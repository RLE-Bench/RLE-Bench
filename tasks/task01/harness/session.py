"""Metered session: the authority on what an agent's run cost.

Task01 is a speed-run. The agent interacts freely up to a budget during development,
then gets ONE terminal evaluation; the score is the fraction of trials it solved,
discounted by how much of the interaction budget it had spent getting there.

Both phases run the same loop -- drive a controller, read why control came back, reset
when the episode is over -- and differ in only two ways: the agent is told it is being
evaluated, and `reset` scores the current trial instead of merely restarting it.

Design points that matter for trust (CLAUDE.md invariant #1):

  * THE AGENT NEVER HOLDS THE ENV. Everything here runs on the daemon side of the
    socket; the agent's controller is invoked as a callback that receives an observation
    and returns an action. So there is no env object in the agent's process to step
    off-meter, and metering cannot be bypassed by tampering with in-process counters.
  * TWO SEPARATE CURRENCIES. Steps taken during the evaluation are NOT charged to the
    interaction budget -- it has its own step ceiling. Mixing them would let a cheap
    evaluation masquerade as cheap learning.
  * EVALUATION SEEDS ARE NEVER SENT TO THE AGENT. Results are reported as aggregates,
    so the agent cannot fit the specific episodes it is graded on.

The env factory is injected so this logic is testable without MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from . import config as C
from . import env as ENV
from . import ledger as L
from . import evaluation as E
from . import obs as _obs

ObsSpec = _obs.ObsSpec  # re-exported for callers of step

EnvFactory = Callable[..., Any]


class Termination:
    """Why the harness stopped applying actions. Internal.

    `step` reports only the EPISODE_ENDING subset to the agent, as `ended`; the rest
    describe an episode that is still alive.
    """

    # -- imposed by the harness (authoritative) ------------------------------
    ENV_SUCCESS = "env_success"            # the task's own success predicate fired
    ENV_DONE = "env_done"                  # env returned done
    HORIZON = "horizon"                    # episode step limit
    SEGMENT_LIMIT = "segment_limit"        # an evaluation trial's remaining ceiling
    BUDGET_EXHAUSTED = "budget_exhausted"  # interaction budget ran out

    # -- nothing stopped it --------------------------------------------------
    STEP_COMPLETE = "step_complete"        # every action was applied; episode alive

    # Reasons that END THE EPISODE. The distinction is what tells the agent whether it
    # may carry on where it left off or must reset. SEGMENT_LIMIT is absent on purpose:
    # a trial's per-call ceiling returns control without ending the trial.
    EPISODE_ENDING = frozenset({ENV_SUCCESS, ENV_DONE, HORIZON, BUDGET_EXHAUSTED})


@dataclass
class SegmentResult:
    """What applying a batch of actions produced. Internal -- `step` reshapes it.

    `success` is the ENVIRONMENT's predicate, never anything the agent asserted.
    """

    steps: int
    reason: str
    terminated_by: str            # "harness" | "agent"
    final: bool
    success: bool
    obs: dict | None = None       # raw simulator observation; shaped only in step()
    info: dict = field(default_factory=dict)
    # True when this episode (development) or trial (evaluation) cannot be continued.
    # Actions stop being applied for reasons that are not the agent's -- the task
    # succeeded, the environment stopped, the horizon was reached -- and the agent has
    # to tell those apart from simply running out of actions to send.
    episode_over: bool = False



def seed_episode(env, seed: int | None):
    """Make the next `reset()` reproducible.

    robosuite sets `self.rng = np.random.default_rng(seed)` ONCE in __init__
    (environments/base.py), and every reset draws layout, style, object instances and
    placements from it. `reset()` itself takes no seed argument -- calling
    `env.reset(seed=...)` raises TypeError -- so per-episode determinism means
    re-seeding that same generator between episodes. This is the documented seed
    pathway applied per episode, not a protocol deviation, and it avoids
    reconstructing the env (~20 s) for every battery episode.
    """
    if seed is not None and hasattr(env, "rng"):
        import numpy as np

        env.rng = np.random.default_rng(int(seed))
    return env.reset()


def episode_instruction(env) -> str | None:
    """The task instruction, which is episode-specific and comes from the simulator.

    `get_ep_meta()["lang"]` interpolates sampled state (e.g. which drawer), so it is
    only meaningful after reset and cannot be baked into a static task description.
    """
    try:
        return env.get_ep_meta().get("lang")
    except Exception:  # noqa: BLE001
        return None


class AgentFacingError(RuntimeError):
    """Base for refusals whose message is written FOR the agent and is safe to send it.

    THIS IS A SECURITY BOUNDARY, not a taxonomy. The service returns the message of
    anything under here verbatim and REDACTS every other exception, because an
    unexpected exception's text is made of private material -- what the simulator chose
    to say about a scene it could not build, what a harness assertion names. Subclassing
    this is therefore a claim that the message contains nothing the agent may not see,
    and each one below is phrased with that in mind.
    """


class SessionClosed(AgentFacingError):
    """Raised when anything is attempted after the session was sealed."""


class DevelopmentClosed(AgentFacingError):
    """Raised when development interaction is attempted after end_development().

    Declaring development finished has to MEAN something, or it is just a comment.
    """


class EpisodeOver(AgentFacingError):
    """Raised when an episode (development) or trial (evaluation) is continued after
    it ended.

    It is gone -- reset() starts the next one. Refusing is deliberate: the alternative
    is resuming from a stale observation as if nothing happened.
    """


class BudgetExhausted(AgentFacingError):
    """Raised when a request would exceed a budget. Never silently truncated:
    an agent must be able to tell 'refused' from 'ran and did nothing'."""


class ConnectionBusy(AgentFacingError):
    """Refused to a second agent connection while one is already open.

    The session is deliberately single-threaded; two clients interleaving on it would
    hand each other's episodes back -- reset() followed by a read of the run state is
    not atomic across connections. One connection at a time; sequential reconnects
    stay free."""


@dataclass
class Budgets:
    interaction_steps: int
    submissions: int


class _NullRecorder:
    """Recorder-shaped no-op, so every hook below can be called unconditionally.

    __getattr__ rather than a list of stubs: a hook added to DebugRecorder later must
    not be able to crash a run that has debugging switched off.
    """

    enabled = False

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


@dataclass
class SessionState:
    steps_used: int = 0
    submissions_used: int = 0
    best_success_rate: float = 0.0
    episodes: int = 0
    # Steps spent in the CURRENT episode. Development had no episode-level accounting
    # at all, so an over-long episode was never ended and the agent could not tell its
    # own segment cap from the harness stopping it.
    episode_steps: int = 0
    history: list[dict] = field(default_factory=list)


def _as_actions(actions) -> list:
    """One action or a sequence of them, always as a list of actions.

    A single action is a sequence of NUMBERS; a batch is a sequence of sequences. Told
    apart by looking at the first element rather than by a flag, so `sim.step(a)` and
    `sim.step([a, a, a])` are both just what they look like.
    """
    ndim = getattr(actions, "ndim", None)
    if ndim is not None:
        return list(actions) if int(ndim) > 1 else [actions]
    if not len(actions):
        # An EMPTY batch is nothing to do, not one empty action. An agent's loop can
        # legitimately compute no actions this time round, and the falling-through
        # `[actions]` below turned that into a malformed 0-component action.
        return []
    if hasattr(actions[0], "__len__"):
        return list(actions)
    return [actions]


class MeteredSession:
    def __init__(
        self,
        task: str,
        budgets: Budgets,
        ledger: L.Ledger,
        env_factory: EnvFactory,
        trial_seeds: Sequence[int] | None = None,
        dev_split: str = "pretrain",
        max_episode_steps: int | None = None,
        eval_plan_fn=None,
        transcript=None,
        recorder=None,
    ):
        if budgets.interaction_steps < 0 or budgets.submissions < 0:
            raise ValueError("budgets must be non-negative")

        self.task = task
        self.budgets = budgets
        self.ledger = ledger
        self._env_factory = env_factory
        self._trial_seeds = list(trial_seeds or [])
        self.dev_split = dev_split
        self.max_episode_steps = max_episode_steps
        # Human-facing record of the evaluation. Optional, and never read by
        # the scorer -- see transcript.py.
        self.transcript = transcript
        # Live operator-facing view, disabled unless asked for. Every call on it is a
        # no-op when off, which is why the hooks below are unconditional. Never read by
        # the scorer, and unreachable by the agent -- see debug.py.
        self.recorder = recorder or _NullRecorder()

        self.state = SessionState()
        self._dev_env = None
        self._closed = False
        self._development_ended = False
        self._episode_over = False
        # Segments continue the CURRENT episode, so the last observation has to
        # survive across run_segment calls.
        self._last_obs: dict | None = None
        self._journalled_in_segment = 0
        # Set while a submission is in progress. Its presence changes what `reset`
        # and `run_segment` mean, which is the whole point of the protocol.
        self._eval: E.EvaluationRun | None = None
        # An evaluation plan is (scene_id, layout, style, trial_seed) per trial.
        # `trial_seeds` is a shorthand for callers that only need one scene: one trial
        # per seed. Exactly one of the two must be supplied.
        if eval_plan_fn is None and not self._trial_seeds:
            raise ValueError(
                "supply eval_plan_fn, or trial_seeds as a single-scene shorthand"
            )
        self._eval_plan_fn = eval_plan_fn or (
            lambda task: [(1, 1, 1, s) for s in self._trial_seeds])

        self.ledger.append(
            L.KIND_START,
            task=task,
            split=dev_split,
            # Evaluation always runs on RoboCasa's own target split -- see
            # evaluation.EvaluationRun._scene_env. Recorded as provenance, not a knob.
            eval_split=E.EVAL_SPLIT,
            interaction_budget=budgets.interaction_steps,
            submission_budget=budgets.submissions,
            planned_trials=len(self._eval_plan_fn(task)),
            # Which way up the agent's camera frames were. Provenance, not a knob: it
            # changes the pixels a run was scored against, so results from either side
            # of the change are not comparable and the ledger should say which side it
            # is on. Runs before this recorded nothing and were bottom-up.
            image_convention=ENV.image_convention(),
        )

    # -- budget accounting ---------------------------------------------------
    @property
    def steps_remaining(self) -> int:
        return max(0, self.budgets.interaction_steps - self.state.steps_used)

    @property
    def submissions_remaining(self) -> int:
        return max(0, self.budgets.submissions - self.state.submissions_used)

    def _charge_steps(self, n: int) -> None:
        if n > self.steps_remaining:
            raise BudgetExhausted(
                f"requested {n} steps but only {self.steps_remaining} remain of "
                f"{self.budgets.interaction_steps}"
            )
        self.state.steps_used += n
        # After the check, not before: a refused charge must not inflate the episode.
        self.state.episode_steps += int(n)

    def _require_open(self) -> None:
        if self._closed:
            raise SessionClosed("session is closed")

    def _live(self) -> bool:
        """Is there something to act in? A development episode, or an evaluation trial.

        The two lifecycles are tracked separately -- an evaluation does not end the
        development episode -- but the agent sees one answer, which is what lets a
        single loop drive both phases.
        """
        if self._eval is not None:
            return self._eval.trial_live
        return not self._episode_over

    def _require_live_episode(self) -> None:
        if not self._live():
            raise EpisodeOver(
                "this episode has ended; call reset() to start the next one"
            )

    def _require_development_open(self) -> None:
        """Development interaction is refused once the agent has declared itself done.

        Only development: evaluation drives the same methods, and by then `_eval` is
        set, so callers check this only on the development path.
        """
        if self._development_ended:
            raise DevelopmentClosed(
                "development is over (end_development was called); "
                "the harness opens the evaluation for the next phase"
            )

    # -- development interaction (charged) -----------------------------------
    def _ensure_dev_env(self):
        if self._dev_env is None:
            self._dev_env = self._env_factory(task=self.task, split=self.dev_split)
        return self._dev_env

    def reset(self, seed: int | None = None) -> dict:
        """Start a new episode.

        During DEVELOPMENT this resets the workspace and arm, and COSTS ONE INTERACTION
        STEP. A reset is a real draw on the simulator -- it rebuilds or re-randomises a
        scene -- and a free one makes abandoning an episode cheaper than finishing it,
        which is the opposite of what a multi-step task should reward.

        During EVALUATION reset is the ONLY thing that advances a trial, and it means
        one of two things depending on what it lands on:

          trial still running   GIVE UP. Scored immediately, in whatever state it is
                                in, and the next trial begins. That is what makes
                                abandoning a trial cost something.
          trial already ended   nothing to forfeit; simply start the next one.

        The harness never advances by itself. If it did, the loop every instruction
        documents -- run a segment, reset when `episode_over` -- would land that reset
        on a freshly-started trial and forfeit it with zero steps.
        """
        self._require_open()
        if self._eval is not None:
            return self._advance_evaluation()
        self._require_development_open()
        return self._open_dev_episode(seed)

    def _open_dev_episode(self, seed: int | None = None) -> dict:
        """Start a development episode and charge the one step it costs.

        THE only way a development episode begins, so the charge cannot be walked
        around by reaching an episode some other way -- `step` opening the first one
        implicitly pays exactly what `reset` would.
        """
        env = self._ensure_dev_env()
        # Charge BEFORE anything is built or any state moves, so an empty budget raises
        # and leaves the session exactly as it was. Also before `episode_steps` is
        # zeroed below -- `_charge_steps` accrues to it, and the reset must not eat the
        # new episode's horizon.
        self._charge_steps(1)
        # Journalled here rather than through `_flush_interaction`, which accrues within
        # one segment: the ledger is the authority the reward is computed from, so it
        # must never lag what was consumed. Resets are rare next to steps, so a record
        # apiece costs nothing.
        self.ledger.append(L.KIND_INTERACT, steps=1)
        obs = seed_episode(env, seed)
        self.state.episodes += 1
        self.state.episode_steps = 0
        self._episode_over = False
        self._last_obs = obs
        # Reset IS the episode boundary, so this closes the previous episode's video
        # and opens the next one.
        self.recorder.episode_started("development", self.state.episodes, obs)
        return obs

    # -- interactive evaluation ---------------------------------------------
    def end_development(self) -> dict:
        """Declare development finished. The agent's own signal that it is ready.

        This is deliberately INERT with respect to the simulator: it starts no
        evaluation, spends no submission, constructs no environment and touches no
        scene. All it does is close the development phase and record that it happened,
        so the interaction cost at the moment the agent judged itself ready is a fact in
        the ledger rather than a claim.

        Entering the scored phase remains the harness's decision -- it happens between
        Harbor steps, not here. Idempotent, so calling it twice is harmless.
        """
        self._require_open()
        if self._development_ended:
            return self._development_summary()
        self._development_ended = True
        self.ledger.append(
            L.KIND_DEV_END,
            steps_used=self.state.steps_used,
            episodes=self.state.episodes,
        )
        return self._development_summary()

    def _development_summary(self) -> dict:
        return {
            "development_ended": True,
            "steps_used": self.state.steps_used,
            "steps_remaining": self.steps_remaining,
            "episodes": self.state.episodes,
        }

    def begin_evaluation(self, plan=None) -> dict:
        """Spend one submission and open an interactive evaluation.

        The agent then drives trials exactly as in development -- run_segment,
        inspect, decide -- and the harness scores each trial when it ends.
        """
        self._require_open()
        if self._eval is not None:
            raise RuntimeError("an evaluation is already in progress")
        if self.submissions_remaining <= 0:
            raise BudgetExhausted(
                f"submission budget exhausted ({self.budgets.submissions} used)"
            )
        self.state.submissions_used += 1

        if plan is None:
            if self._eval_plan_fn is None:
                raise RuntimeError("no evaluation plan configured")
            plan = self._eval_plan_fn(self.task)

        self._eval = E.EvaluationRun(
            task=self.task,
            plan=tuple(plan),
            env_factory=self._env_factory,
            max_steps_per_trial=self.max_episode_steps,
        )
        first = self._eval.start_trial()
        self._log_trial_start(first)
        self.recorder.episode_started("evaluation", 0, self._eval._obs)
        return {
            "submission_index": self.state.submissions_used - 1,
            "trial": first,
            "total_trials": self._eval.total_trials,
        }

    def _log_trial_start(self, descriptor: dict | None) -> None:
        run = self._eval
        if self.transcript is None or run is None or descriptor is None:
            return
        entry = run.plan[descriptor["position"]]
        try:
            self.transcript.trial_start(
                trial=descriptor["position"], scene=entry[0], layout=entry[1],
                style=entry[2], seed=entry[3], env=run._env, obs=run._obs)
        except Exception:  # noqa: BLE001
            pass   # the record is diagnostic; it must never break a trial

    def _end_trial(self, ended_by: str, error: str | None = None):
        """End and score the current trial, and record it for a human reader.

        Both the ledger-facing and human-facing sides go through here so the two
        cannot disagree about how a trial finished.
        """
        run = self._eval
        assert run is not None
        env, obs = run._env, run._obs
        rec = run.end_trial(ended_by, error=error)
        self.recorder.diagnosis(env, obs, ended_by=ended_by, success=rec.success,
                                steps=rec.steps, trial=len(run.results) - 1)
        self.recorder.event("trial_end", ended_by=ended_by, success=rec.success,
                            steps=rec.steps, trial=len(run.results) - 1)
        if self.transcript is not None:
            try:
                self.transcript.trial_end(
                    trial=len(run.results) - 1, ended_by=ended_by,
                    success=rec.success, steps=rec.steps, env=env, obs=obs)
            except Exception:  # noqa: BLE001
                pass
        return rec

    def _advance_evaluation(self) -> dict:
        """The evaluation half of `reset`. See reset() for what the two cases mean."""
        run = self._eval
        assert run is not None
        if run.trial_live:
            self._end_trial(E.TRIAL_END_RESET)
        return self._advance_or_finish()

    def _advance_or_finish(self) -> dict:
        run = self._eval
        assert run is not None
        if run.current() is None:
            return self._finish_evaluation()
        nxt = run.start_trial()
        if nxt is None:
            return self._finish_evaluation()
        self._log_trial_start(nxt)
        self.recorder.episode_started("evaluation", run.position, run._obs)
        return {"trial": nxt, "evaluation_done": False}

    def _finish_evaluation(self) -> dict:
        run = self._eval
        assert run is not None
        summary = run.finish()
        if self.transcript is not None:
            try:
                self.transcript.summary(summary)
            except Exception:  # noqa: BLE001
                pass
        self._eval = None
        self.state.best_success_rate = max(self.state.best_success_rate,
                                           summary["success_rate"])
        self.ledger.append(
            L.KIND_SUBMIT,
            submission_index=self.state.submissions_used - 1,
            steps_cumulative=self.state.steps_used,
            success_rate=summary["success_rate"],
            episodes=summary["trials"],
            trials=summary["trials"],
            # How many trials the agent actually reached. Diagnosis only -- it never
            # enters the reward -- but it is what tells a zero from a broken run.
            trials_attempted=summary["trials_attempted"],
            eval_steps=summary["steps_used"],
            per_scene=run.per_scene(),
            controller_sha256=None,
        )
        # Aggregates only: per-scene and per-trial detail stays in the ledger, since
        # returning it would let the agent localise the graded episodes.
        result = {
            "evaluation_done": True,
            "submission_index": self.state.submissions_used - 1,
            "success_rate": summary["success_rate"],
            "trials": summary["trials"],
            "trials_attempted": summary["trials_attempted"],
            "submissions_remaining": self.submissions_remaining,
            "steps_remaining": self.steps_remaining,
        }
        self.state.history.append(result)
        return result

    @property
    def evaluating(self) -> bool:
        return self._eval is not None

    @property
    def current_env(self):
        """The env an observation would come from right now, or None.

        The development env, or -- once the evaluation is open -- the live trial's. One
        accessor rather than two call sites reaching for `_dev_env` and `_eval._env`,
        because a caller that picked the wrong one would read the right-shaped state
        from the wrong scene and nothing would say so.

        Never BUILDS one: this is asked on the observation path, where an env that does
        not exist yet means there is nothing to describe, not that one should be made.
        """
        if self._eval is not None:
            return self._eval._env
        return self._dev_env

    @property
    def phase(self) -> str:
        """Which phase the run is in. The agent must always be able to tell, because
        during evaluation `reset` scores the trajectory instead of just restarting.
        "awaiting_evaluation" is the gap between `end_development` and the harness
        opening the submission. "finished" is terminal: development is closed and no
        submission remains, so nothing can be driven again."""
        if self._eval is not None:
            return "evaluation"
        if self._development_ended:
            return ("finished" if self.submissions_remaining <= 0
                    else "awaiting_evaluation")
        return "development"

    def trial_info(self) -> dict | None:
        """Descriptor of the trial currently in progress, or None outside evaluation.

        Needed because the harness opens the evaluation before the agent starts (so the
        phase is not the agent's choice), which means the agent never sees the reply to
        begin_evaluation and has to ask for the trial it has been dropped into.
        """
        run = self._eval
        # None once the evaluation is over. While it is running there is always a live
        # trial: the harness starts the next one as soon as the previous is scored.
        if run is None:
            return None
        return run.descriptor()

    def observe(self, spec=None) -> dict:
        """The current observation, without advancing anything. FREE.

        Perception is deliberately an explicit act here rather than something that only
        arrives as a side effect of stepping. Looking costs a real robot nothing but
        time, and an agent should not have to spend budget -- or worse, take an action
        it does not want -- merely to see where it is.

        Free in the interaction sense: no env.step, no charge, no ledger record. It
        still costs wall clock when a re-render is asked for, and the phase clock is
        what bounds that.

        `spec` is an ObsSpec, resolved by the same function that shapes the
        observations streamed to a running controller -- so what you look at and what
        you act on cannot disagree.

        `live` says whether there is an episode to act in. An observation of a finished
        one comes back thinner than it was asked for, and without this the reply could
        not be told apart from a working one -- a caller that had asked for depth met
        that as a `KeyError` several frames later. Testing it is free, and it is the
        same answer `step` would raise `EpisodeOver` on.
        """
        self._require_open()
        run = self._eval
        if run is not None:
            env, obs = run._env, run._obs
        else:
            env = self._ensure_dev_env()
            obs = self._last_obs
            # Nothing to look at: no episode has been opened, or the last one ended.
            # `observe` must NOT reset to manufacture something -- a reset costs a step,
            # and one taken here would be both unmetered and invisible, moving the
            # episode boundary the agent never asked to move.
        if obs is None or env is None:
            return {"obs": {}, "instruction": None, "resolution": None, "live": False}

        obs, resolution = ENV.apply_obs_spec(
            env, dict(obs), spec, default=C.OBS_RESOLUTION,
            rendered=C.RENDER_RESOLUTION, ceiling=C.OBS_MAX_RESOLUTION)
        return {
            "obs": obs,
            "instruction": episode_instruction(env),
            "resolution": resolution,
            "live": self._live(),
        }

    def step(self, actions, obs_spec=None) -> dict:
        """Apply actions in the current episode or trial. THE ONLY WAY TO ACT.

        `actions` is one action or a sequence of them, applied in order and stopping at
        the first that ends the episode -- so the reply's `steps` is what was APPLIED,
        not what was asked for. Identical in both phases; only the currency differs
        (development charges the interaction budget, evaluation the trial's ceiling).

        Returns {"obs", "steps", "success", "episode_over", "ended", "steps_remaining"}.
        `success` is the ENVIRONMENT's own predicate and nothing else; `ended` says why
        the episode stopped and is None while it is alive.

        There is nothing to implement and nothing to hand over: write whatever loop you
        like around this, in your own code, deciding each move with the observation in
        front of you.
        """
        res = self._act(_as_actions(actions))
        # Specs describe this reply, never the observation retained by the harness.
        shown, _ = ENV.apply_obs_spec(
            self.current_env, res.obs, obs_spec, default=C.OBS_RESOLUTION,
            rendered=C.RENDER_RESOLUTION, ceiling=C.OBS_MAX_RESOLUTION)
        return {
            "obs": shown,
            "steps": res.steps,
            "success": res.success,
            "episode_over": res.episode_over,
            # The agent is told WHY, but only for reasons that are facts about the
            # episode. `step_complete` means "your actions ran out", which it knows.
            "ended": res.reason if res.reason in Termination.EPISODE_ENDING else None,
            "steps_remaining": self._step_figures()[1],
            # Diagnostics: which cap ended a graded trial, or what an environment
            # error was. `episode_over` remains the authoritative "you must reset".
            "info": dict(res.info or {}),
        }



    def _act(self, actions) -> SegmentResult:
        """Apply actions in the CURRENT episode or trial. No reset.

        That is what lets the agent work in pieces: send a few actions, look at what
        came back, send the next few, all inside one episode. Every action is charged.
        """
        self._require_open()
        if self._eval is not None:
            # A scored trial is never resumed or silently advanced past: only reset()
            # starts the next trial, so a controller stepping blindly cannot land in a
            # trial it was never told about.
            if not self._eval.trial_live:
                pos = self._eval.position
                nxt = (f"begin trial {pos}" if self._eval.current() is not None
                       else "close the evaluation")
                raise EpisodeOver(f"trial {pos - 1} scored; call reset() to {nxt}")
            return self._act_eval(actions)
        # BEFORE the liveness check, because exhausting the budget is itself what ended
        # the episode: reporting "call reset()" there would send the agent round a loop
        # that cannot help it, instead of telling it the run is over.
        if self.steps_remaining <= 0:
            env = self._ensure_dev_env()
            return self._segment(0, Termination.BUDGET_EXHAUSTED, "harness", True,
                                 env, self._last_obs, charge=False)
        # Development: continuing an episode that has ended is refused, never resumed.
        self._require_live_episode()
        self._require_development_open()
        env = self._ensure_dev_env()
        # Stepping before the first reset opens an episode, and pays for it: the charge
        # belongs to starting one, not to the call that happened to start it.
        obs = self._last_obs if self._last_obs is not None else self._open_dev_episode()
        # No cap: a development episode is ended by the horizon, which reports HORIZON.
        return self._apply(env, actions, obs, None, charge=True)

    def _act_eval(self, actions) -> SegmentResult:
        """Actions inside an evaluation trial.

        Steps are charged to the SUBMISSION, never to the interaction budget. When the
        environment stops or the trial runs out of steps, the trial is scored here, so
        the agent drives a graded trial with exactly the loop it used in development.
        """
        run = self._eval
        assert run is not None
        env = run._env
        cap = run.steps_left_in_trial()

        trial_error: str | None = None
        try:
            res = self._apply(env, actions, run._obs, cap, charge=False)
        except Exception as exc:  # noqa: BLE001
            # The ENVIRONMENT raised. There is no way to keep driving this trial, but
            # the trials behind it are still worth running, so it is scored as a lost
            # trial rather than allowed to abort the evaluation.
            trial_error = f"{type(exc).__name__}: {exc}"
            res = SegmentResult(
                steps=0, reason=Termination.ENV_DONE, terminated_by="harness",
                final=True, success=False, obs=run._obs,
                info={"error": f"env: {trial_error}"})
        else:
            run.charge(res.steps)
            run._obs = res.obs if res.obs is not None else run._obs

        if self.transcript is not None:
            try:
                self.transcript.segment(len(run.results), run.segment_index, res)
                run.segment_index += 1
            except Exception:  # noqa: BLE001
                pass

        # The trial ends when the environment stops, when it errored, or when it has no
        # steps left.
        ended = None
        if trial_error is not None:
            ended = E.TRIAL_END_ERROR
        elif res.reason == Termination.ENV_SUCCESS:
            ended = E.TRIAL_END_ENV_SUCCESS
        elif res.reason == Termination.ENV_DONE:
            ended = E.TRIAL_END_ENV_DONE
        elif (run.steps_left_in_trial() or 0) <= 0:
            ended = E.TRIAL_END_MAX_STEPS
        if ended is not None:
            self._end_trial(ended, error=trial_error)
            res.info = dict(res.info or {})
            res.info["trial_ended"] = ended
            # Authoritative: the trial is gone.
            res.episode_over = True
            # A trial that ran out of steps reports it as `ended`, the same way a
            # development episode reports its horizon; `info` keeps which cap it was.
            if ended == E.TRIAL_END_MAX_STEPS:
                res.reason = Termination.HORIZON
        return res

    @staticmethod
    def _safe_success(env) -> bool:
        """The env's own success predicate, which can itself raise on a broken env."""
        try:
            return bool(env._check_success()) if env is not None else False
        except Exception:  # noqa: BLE001
            return False

    def _apply(self, env, actions, obs, cap, *, charge: bool) -> SegmentResult:
        """Apply actions in order on `env`, stopping at the first that ends the episode.

        THE ONE LOOP. Both phases go through it; `charge=False` for evaluation, where
        trial steps are a submission cost and must not touch the interaction budget.

        Nothing here calls back into the agent. Actions arrive already validated and
        already decided, which is what makes the agent's own loop -- send some actions,
        look at what came back, decide the next ones -- the whole interface.

        The raw observation is kept for the harness's own use (the success predicate,
        the debug recorder): narrowing what the agent is shown must never narrow what
        the harness checks.
        """
        steps = 0
        self._journalled_in_segment = 0
        for action in actions:
            # The trial's remaining ceiling, in evaluation. Development has no caller
            # cap, so the horizon below is what ends an episode there.
            if cap is not None and steps >= cap:
                return self._segment(steps, Termination.SEGMENT_LIMIT, "harness",
                                     False, env, obs, charge=charge)
            if (charge and self.max_episode_steps is not None
                    and self.state.episode_steps >= self.max_episode_steps):
                return self._segment(steps, Termination.HORIZON, "harness",
                                     True, env, obs, charge=charge)
            if charge and self.steps_remaining <= 0:
                return self._segment(steps, Termination.BUDGET_EXHAUSTED, "harness",
                                     True, env, obs, charge=charge)

            obs, _reward, done, _i = env.step(action)
            steps += 1
            self.recorder.frame(obs)
            if charge:
                self._charge_steps(1)
                self._last_obs = obs

            if env._check_success():
                return self._segment(steps, Termination.ENV_SUCCESS, "harness",
                                     True, env, obs, charge=charge)
            if done:
                return self._segment(steps, Termination.ENV_DONE, "harness", True,
                                     env, obs, charge=charge)

        # Everything asked for was applied and the episode is still alive.
        return self._segment(steps, Termination.STEP_COMPLETE, "agent", False,
                             env, obs, charge=charge)

    def _segment(self, steps, reason, by, final, env, obs, extra=None,
                 charge: bool = True) -> SegmentResult:
        if charge:
            self._flush_interaction(steps)
        reason = reason or Termination.STEP_COMPLETE
        over = reason in Termination.EPISODE_ENDING
        if charge and over:
            # Evaluation liveness is tracked separately by _act_eval.
            self._episode_over = True
            self._last_obs = None
            self.recorder.diagnosis(env, obs, ended_by=reason,
                                    episode=self.state.episodes)
        result = SegmentResult(
            steps=steps, reason=reason,
            terminated_by=by, final=bool(final),
            success=bool(env._check_success()), obs=obs, info=dict(extra or {}),
            episode_over=over,
        )
        self.recorder.event("segment", steps=steps, reason=reason, terminated_by=by,
                            success=result.success, episode_over=over,
                            steps_used=self.state.steps_used)
        return result

    def _flush_interaction(self, steps: int) -> None:
        """Journal accrued interaction. Guarded so the segment paths, which may
        flush and then build a result, cannot double-count the same steps."""
        pending = int(steps) - self._journalled_in_segment
        if pending > 0:
            self.ledger.append(L.KIND_INTERACT, steps=pending)
            self._journalled_in_segment = int(steps)

    # -- teardown ------------------------------------------------------------
    def close(self, end_reason: str = "harness_seal") -> dict:
        """Seal the ledger. A missing seal means the run was cut short before its
        results were written, and the verifier treats it as unscoreable.

        `end_reason` records HOW the run ended -- an explicit agent close, the phase
        clock expiring, the daemon being signalled after the agent exited. It is
        diagnosis only and never reaches the reward, but without it every run looks
        alike in the ledger and an infrastructure failure is indistinguishable from a
        genuine zero.
        """
        if self._closed:
            return L.summarize(L.load_verified(self.ledger.path))
        # An evaluation that was opened but never finished must still be SCORED. The
        # harness opens it between steps, so a run whose agent never starts -- a crash,
        # a timeout, a failed resume -- would otherwise seal with the submission spent
        # and no submit record at all: the seal said submissions_used=1 while the record
        # stream said 0, and the planned trials were never marked failed. finish()
        # already scores unreached trials as failures; it simply was not called.
        if self._eval is not None:
            self._finish_evaluation()
        closer = getattr(self._dev_env, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass          # a stuck env must not stop the ledger being sealed
        # record_count includes the seal itself.
        count = sum(1 for _ in L.read_records(self.ledger.path)) + 1
        self.ledger.append(
            L.KIND_SEAL,
            record_count=count,
            # Digest of the human-facing transcript. Recorded here so that editing the
            # transcript -- which lives in an agent-writable artifacts dir -- is
            # detectable, even though the transcript is never scored.
            transcript_sha256=(self.transcript.digest()
                               if self.transcript is not None else None),
            steps_used=self.state.steps_used,
            submissions_used=self.state.submissions_used,
            best_success_rate=self.state.best_success_rate,
            end_reason=end_reason,
        )
        self._closed = True
        self.recorder.close()
        return L.summarize(L.load_verified(self.ledger.path))

    def probe(self) -> dict:
        """Construct the dev env and learn its action contract, without charging.

        `env.action_spec` is only valid after the first reset, so the daemon does one
        free reset at startup; otherwise the agent's first `task_info()` would report
        action_dim=None.
        """
        env = self._ensure_dev_env()
        obs = env.reset()
        self.state.episodes += 1
        low, _ = env.action_spec
        return {
            "action_dim": int(len(low)),
            "instruction": episode_instruction(env),
            "obs_keys": sorted(obs.keys()),
        }

    def _step_figures(self) -> tuple:
        """(used, remaining), phase-local: the interaction budget in development, the
        open trial's count and remaining ceiling in evaluation -- the only budget an
        evaluation agent can actually spend."""
        run = self._eval
        if run is None:
            return self.state.steps_used, self.steps_remaining
        return run.trial_steps, run.steps_left_in_trial()

    def status(self) -> dict:
        """What the agent is allowed to know about its own run."""
        used, remaining = self._step_figures()
        return {
            "task": self.task,
            "phase": self.phase,
            "steps_used": used,
            "steps_remaining": remaining,
            "submissions_used": self.state.submissions_used,
            "submissions_remaining": self.submissions_remaining,
        }
