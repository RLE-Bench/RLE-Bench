"""Every task01 default lives here. Nothing numeric is hardcoded elsewhere.

Per CLAUDE.md: thresholds live in one place, never scattered across files. Each value
below is the DEFAULT; the ones a deployment is expected to tune are overridable from
`task.toml` (see `tasks/task01/README.md` for the table of knobs and which env channel
each one travels on).

Reference anchors for the numbers here:
  step        ~38 ms      (GPU EGL, cameras essentially free)
  reset       ~4.8 s      steady state; the first reset of a process costs ~17 s
  horizon     1000        RoboCasa default, kept by decision
"""

from __future__ import annotations

import os
import re

import hashlib


# --- reading overrides out of the environment --------------------------------
# `task.toml` has no generic task-parameter mechanism, so every knob a deployment can
# turn travels as an environment variable. Two channels, and which one a knob uses is a
# decision about who may read it:
#
#   [environment.env]  reaches the whole container -- INCLUDING the agent, which can
#                      read its own /proc/self/environ. Fine for anything the agent is
#                      told anyway (its budgets, its clocks).
#   [verifier.env]     reaches the verifier's exec environment only. This is where
#                      scoring weights belong.
#
# A THIRD CASE HAS NO ENV CHANNEL AT ALL. Values the DAEMON needs but the agent must not
# see -- the evaluation plan and the seed salt -- travel by a root-only FILE
# (`eval_overrides`, below), never by a variable. The obvious-looking alternative does
# not work: setting them container-wide and unsetting them in entrypoint.sh before the
# agent is exec'd leaves them fully readable, because `docker exec` builds each new
# process's environment from the container's stored config rather than from PID 1's live
# environ, and Harbor starts every agent with `docker exec`.
#
# Malformed values raise rather than falling back silently. A run that quietly ignored
# its own configuration would be scored under rules nobody chose.
def _env(name: str, default, cast, kind: str):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise ValueError(f"{name} must be {kind}, got {raw!r}") from None


def env_int(name: str, default: int) -> int:
    return _env(name, default, int, "an integer")


def env_float(name: str, default: float) -> float:
    return _env(name, default, float, "a number")


# --- the harness level -------------------------------------------------------
# WHAT THE AGENT IS GIVEN, which is the axis task01 varies. The task, the scoring and
# the evaluation plan are identical at every level; only the harness differs, so a
# score at L1 and a score at L3 are directly comparable.
#
#   L1  the low-level controller API and nothing else: three cameras, proprioception,
#       and a 12-D action. Where things are has to be perceived. THE CONTROL
#       CONDITION.
#   L2  L1 plus the harness library and its manual, shipped world-readable in the
#       image. Its skills are CLIENT-SIDE, so every step they take is metered exactly
#       as a hand-written one would be, and a skill can see nothing an L1 agent could
#       not.
#   L3  L2 plus privileged simulator state -- the poses agent_visible_obs strips out.
#       See tasks/task01/harness/privileged.py, which ships only in the root-only tree.
#       L3 carries the library too, so L3 minus L2 isolates the ONE thing that differs
#       between them; giving L3 the state but not the tools would confound "perception
#       is free" with "you have no tools", which is two changes, not one.
#
# BUILD-TIME, not run-time, and that is what the level string here records rather than
# decides: which modules land in the agent-readable payload is settled by
# tasks/task01/build_assets.py when the image is built. This value tells the DAEMON
# whether to attach privileged state, and is reported to the agent (it is told its
# level) and written to the ledger's start record, so a run is always attributable to
# the harness it actually ran under.
LEVELS = ("L1", "L2", "L3")

# The level that ships privileged state. Named rather than compared inline, so the one
# place the daemon widens the observation boundary is greppable.
PRIVILEGED_LEVEL = "L3"

# The levels that ship the harness library. Read by build_assets.py, never by the
# daemon: skills run in the agent's own process and the daemon cannot tell one from any
# other controller -- which is the point, and which is why this is a build-time fact
# about the image rather than anything the wire knows.
SKILLS_LEVELS = ("L2", "L3")


def level() -> str:
    """The harness level this container was built for.

    Malformed values raise rather than defaulting to L1: a run that quietly graded the
    wrong harness would be worse than one that failed to start.
    """
    raw = os.environ.get("RLEBENCH_LEVEL", "").strip().upper() or LEVELS[0]
    if raw not in LEVELS:
        raise ValueError(
            f"RLEBENCH_LEVEL must be one of {', '.join(LEVELS)}, got {raw!r}")
    return raw


# --- the subtasks ------------------------------------------------------------
# The RoboCasa ATOMIC tasks task01 emits a Harbor task directory for, at each level.
#
# NONE OF THIS IS BAKED INTO AN IMAGE. The task name reaches the daemon as
# `--task` (daemon_main) and is a pure runtime string from there on -- make_env takes
# it, eval_trials derives seeds from it, and nothing else is task-specific. So the
# directories at a level share ONE image and differ only in RLEBENCH_TASK. task02 does
# the same with RLEBENCH_GROUP; its held-out names stay in a root-only table.
#
# Articulations, a knob turn and pick-and-places, so a level's effect can be read against
# task difficulty rather than against one task. Slugs are numbered by position here.
SUBTASKS: tuple[tuple[str, str], ...] = (
    ("OpenFridge", "open the fridge door"),
    ("CloseCabinet", "close the cabinet door"),
    ("TurnOnStove", "turn on the stove burner"),
    ("PickPlaceCounterToDrawer", "move an object from the counter into the drawer"),
    ("PickPlaceMicrowaveToCounter", "move an object from the microwave onto the counter"),
)


def subtask_slug(index: int, task: str) -> str:
    """The emitted directory name for a subtask: `01-open-fridge`.

    Numbered so the ladder order survives an alphabetical listing, which is how both
    the Makefile and the suite script enumerate them.
    """
    import re as _re

    kebab = _re.sub(r"(?<!^)(?=[A-Z])", "-", task).lower()
    return f"{index:02d}-{kebab}"


# --- budgets ----------------------------------------------------------------
# 100k steps ~ 63 min of pure stepping at 38 ms. Deliberately finite: the speed-run
# measures how much simulator experience is burned before the agent declares itself
# ready, so the cap has to bite. Roughly 100 full-horizon episodes' worth.
#
# Override with RLEBENCH_INTERACTION_STEPS. The scorer reads the budget back out of
# the ledger's start record rather than from here, so a per-run override rescales the
# efficiency term instead of invalidating it.
INTERACTION_STEPS = 100_000
# ONE terminal submission: the agent gets a single irreversible call on whether it is
# ready.
SUBMISSIONS = 1

# --- evaluation structure ---------------------------------------------------
# Development and evaluation share one agent session. end_development opens the
# hidden plan; reset is then unavailable. The agent can keep editing its controllers.
#
# The plan is pairs of (scene_id, trials_in_that_scene). Scene ids index
# TARGET_SCENE_IDS, the target split's own (layout, style) pairs, so pinning a scene
# stays inside RoboCasa's protocol rather than inventing a distribution. Trials within a
# scene vary object instances and placements.
#
# Default: five scenes, ONE trial each. Every trial is a distinct (layout, style) pair,
# so all five measure generalisation across scenes rather than luck within one; the
# success rate lands on 0.2 increments, which still has a middle to rank agents by.
# Repeating a scene would buy finer granularity at the cost of spending trials on a
# kitchen the agent has already seen.
#
# Override from the ROOT-ONLY file below, e.g. "1x2,2x2,3x2" (see parse_eval_plan).
EVAL_PLAN: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (3, 1), (4, 1), (5, 1))

# The target split's scene pairs. Scene ids are 1-based, matching RoboCasa's layout and
# style numbering.
TARGET_SCENE_IDS = tuple((i, i) for i in range(1, 11))

# WHERE THE PLAN AND THE SALT COME FROM. A file inside /opt/private, which is root:root
# 0700 -- the agent cannot traverse into it, so unlike an environment variable this
# cannot be read out of /proc/self/environ. Absent by default, which is the normal case:
# the constants above are then the plan, and the salt is empty.
EVAL_OVERRIDES_PATH = "/opt/private/eval_plan.txt"


def eval_overrides(path: str | None = None) -> dict[str, str]:
    """`KEY=VALUE` lines from the root-only override file; {} when there is none.

    `path` defaults to EVAL_OVERRIDES_PATH, resolved at CALL time rather than bound as a
    default argument, so a test can repoint the constant.

    Recognised keys are `plan` and `salt`. Unknown keys RAISE rather than being ignored:
    a typo'd key in a file only an operator can write means the run is about to be
    graded under a plan nobody chose, and silence is the wrong answer to that.
    """
    path = path or EVAL_OVERRIDES_PATH
    try:
        with open(path) as f:
            raw = f.read()
    except FileNotFoundError:
        return {}
    out: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"{path}: expected KEY=VALUE, got {line!r}")
        key = key.strip().lower()
        if key not in ("plan", "salt"):
            raise ValueError(f"{path}: unknown key {key!r}; expected 'plan' or 'salt'")
        out[key] = value.strip()
    return out


def parse_eval_plan(spec: str) -> tuple[tuple[int, int], ...]:
    """Parse an evaluation plan from its `task.toml` string form.

    "<scene>x<trials>" entries, comma separated: "1x2,2x2,3x2" is three scenes with two
    trials each. Validation is deliberately eager -- a malformed plan must fail at
    daemon start, where the message is visible, rather than at the first trial.
    """
    entries: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        scene, sep, trials = chunk.lower().partition("x")
        if not sep:
            raise ValueError(
                f"malformed evaluation plan entry {chunk!r}: expected '<scene>x<trials>'"
            )
        try:
            entry = (int(scene.strip()), int(trials.strip()))
        except ValueError:
            raise ValueError(
                f"malformed evaluation plan entry {chunk!r}: scene and trials must be "
                f"integers"
            ) from None
        entries.append(entry)
    if not entries:
        raise ValueError("evaluation plan is empty")
    plan = tuple(entries)
    # Reuse the plan validation that already lives in eval_trials rather than
    # duplicating the bounds checks here.
    eval_trials("__plan_check__", plan=plan)
    return plan


def plan_trial_count(plan: tuple[tuple[int, int], ...] = EVAL_PLAN) -> int:
    return sum(int(t) for _, t in plan)


def eval_trials(task: str, plan: tuple[tuple[int, int], ...] = EVAL_PLAN,
                salt: str | None = None) -> tuple[tuple[int, int, int, int], ...]:
    """Expand an evaluation plan into per-trial entries.

    Returns (scene_index, layout_id, style_id, trial_seed) per trial. Trial seeds are
    derived per (scene, trial) from the task name, so the plan is reproducible for a
    given build while remaining underivable by the agent, which never sees this module.
    """
    if not plan:
        raise ValueError("evaluation plan must contain at least one scene")
    entries = []
    for scene_id, trials in plan:
        if trials <= 0:
            raise ValueError(f"scene {scene_id} must have a positive trial count")
        if not (1 <= scene_id <= len(TARGET_SCENE_IDS)):
            raise ValueError(
                f"scene id {scene_id} is outside the target split's scenes "
                f"(1..{len(TARGET_SCENE_IDS)})"
            )
        layout, style = TARGET_SCENE_IDS[scene_id - 1]
        for t_i in range(trials):
            h = hashlib.sha256(
                f"{_EVAL_DOMAIN}|{salt or ''}|{task}|scene{scene_id}|trial{t_i}".encode()
            ).digest()[:8]
            entries.append((scene_id, layout, style,
                            int.from_bytes(h, "big") % _SEED_MODULUS))
    return tuple(entries)


# --- reward weights ---------------------------------------------------------
#     reward = W_OUTCOME * sr  +  W_EFFICIENCY * sr * (1 - dev_steps / budget)
#
# `sr` is the fraction of evaluation trials the environment's own success predicate
# accepted. Outcome dominates because solving the task is the point; efficiency is the
# speed-run axis.
#
# Efficiency is SCALED BY `sr` rather than added independently. An independent term
# would pay its full weight to a run that never interacted and never succeeded -- the
# cheapest possible run collecting credit for having done nothing. Scaling makes the
# efficiency axis a multiplier on real progress: a run that solves nothing scores zero
# however little it spent, and among runs that solve the same amount the cheaper one
# wins.
#
# The two weights sum to 1.0, so a perfect run that spent nothing scores exactly 1.0.
# Override with RLEBENCH_W_OUTCOME / RLEBENCH_W_EFFICIENCY, which travel on
# `[verifier.env]` and so are not readable by the agent (see scoring.reward_weights).
W_OUTCOME = 0.80
W_EFFICIENCY = 0.20

# --- cost bounds ------------------------------------------------------------
# The one step cap: bounds every development episode and every graded trial, so a
# submission is bounded at this x the plan's trial count. Set to RoboCasa's own
# default horizon (1000). The harness enforces it itself: create_env passes
# ignore_done=True, so the environment never signals termination.
#
# Override with RLEBENCH_MAX_STEPS_PER_TRIAL.
MAX_STEPS_PER_TRIAL = 1_000


# --- observation resolution --------------------------------------------------
# The per-step render size, and the ceiling `observe()` may ask for.
#
# Two numbers rather than one because they trade against different things. Every
# `step()` renders all three cameras, so OBS_RESOLUTION is on the hot path and stays
# small. `observe()` is off the hot path -- it advances nothing and is charged nothing --
# so it may ask for more detail, bounded by OBS_MAX_RESOLUTION.
#
# A request above the ceiling is clamped rather than refused.
OBS_RESOLUTION = 128
OBS_MAX_RESOLUTION = 512

# EVERY camera is baked at the ceiling, and nothing ever asks MuJoCo for more. Growing
# the offscreen framebuffer mid-run is not a resize: robosuite frees the LIVE GL context
# and builds a new one (`binding_utils.update_offscreen_size`). That rebuild is the only
# operation in these harnesses that has ever logged `OpenGL error 0x501 in or before
# mjr_makeContext`, and in task02 a context that came back broken killed the daemon
# inside the render once and wedged it at 100% CPU once, losing the trials behind it.
# Baking at the ceiling deletes the path: a SMALLER render never resizes the buffer, and
# a smaller delivery is a resample of a frame already in hand.
RENDER_RESOLUTION = OBS_MAX_RESOLUTION

# --- tasks whose success predicate is degenerate ----------------------------
# PrepareBroilingStation's success predicate is already true at reset, so a do-nothing
# controller scores 100% on it. Excluded rather than special-cased, because a task that
# is free to "solve" measures nothing.
EXCLUDED_TASKS = frozenset({"PrepareBroilingStation"})

# --- seed derivation --------------------------------------------------------
# Trial seeds are DERIVED, never published: the agent never sees this module (it ships
# only in the root-only tree) and a submission reports aggregates only, so the plan
# cannot be enumerated. The salt lets a deployment keep its trials private even if the
# code is public.
# FROZEN: the literal string feeds the hidden eval seeds; renaming it re-rolls every seed.
_EVAL_DOMAIN = "rlebench.task01.eval"

_SEED_MODULUS = 2**31 - 1


def is_scoreable(task: str) -> bool:
    return task not in EXCLUDED_TASKS


# -- what the agent is allowed to observe -------------------------------------
#
# IMAGES AND PROPRIOCEPTION ONLY. RoboCasa's observation dict also carries ground-truth
# poses for every object and fixture in the scene -- drawer_obj_pos, drawer_obj_quat,
# drawer_obj_to_robot0_eef_pos, distr_counter_*_pos ... -- which a real robot could only
# obtain through perception and localisation. Handing them to the agent turns a
# perception problem into arithmetic: a run can be solved by reading the drawer's world
# pose and subtracting, never looking at a pixel.
#
# The rule is "keep robot<N>_*", which is exact rather than approximate here, verified
# against a live env:
#   * `object-state` is a concatenation of precisely the object keys, so it must go too
#     -- dropping the individual keys and keeping this would leak all of them;
#   * `robot0_proprio-state` contains NO object information (checked: drawer_obj_pos
#     does not appear in it), so it stays.
#
# Ego-state in the world frame (robot0_base_pos/quat, robot0_eef_pos/quat) is kept: it
# is the robot's own pose, the analogue of odometry, and it is useless for finding an
# object whose position is no longer given. Actions are expressed in the base frame, and
# robot0_base_to_eef_* gives the end-effector in that same frame.
_AGENT_OBS_RE = re.compile(r"^robot\d+_")


def agent_visible_obs(obs):
    """Filter an observation down to what the agent may see.

    Applied at the WIRE, in the daemon: the harness keeps the full observation for
    success checking and for the human-facing transcript, and only the agent's view is
    narrowed. Non-dict input is passed through untouched.
    """
    if not isinstance(obs, dict):
        return obs
    return {k: v for k, v in obs.items() if _AGENT_OBS_RE.match(k)}
