# Task 02 — RoboCasa harness engineering

Can an agent build perception primitives, controllers and a manual that a
**different agent, with no shared context**, can apply to a task neither has seen?

Each activity group has three training tasks and one held-out composite task.
The training set was selected to cover the held-out task's atomic primitives.
There are fifteen groups, each a Harbor task, sharing one simulator image.

```text
develop   agent A practises and writes /workspace/agent_harness/
          MANUAL.md, perception/, controllers/, dependencies
eval_01   fresh agent B reads those files and attempts the held-out task
...
eval_05   fresh agent F receives the same frozen files and an independent scene
```

| Band | Slug | Held-out task |
| --- | --- | --- |
| EASY | `01-washing-dishes` | `DumpLeftovers` |
| EASY | `02-sauteing-vegetables` | `PlaceVegetablesEvenly` |
| EASY | `03-baking` | `PastryDisplay` |
| EASY | `04-reheating-food` | `SimmeringSauce` |
| EASY | `05-chopping-food` | `ClearCuttingBoard` |
| MEDIUM | `06-setting-the-table` | `AlignSilverware` |
| MEDIUM | `07-portioning-meals` | `PortionHotDogs` |
| MEDIUM | `08-defrosting-food` | `DefrostByCategory` |
| MEDIUM | `09-arranging-buffet` | `PlaceBeveragesTogether` |
| MEDIUM | `10-serving-beverages` | `MatchCupAndDrink` |
| HARD | `11-loading-fridge` | `MoveFreezerToFridge` |
| HARD | `12-managing-freezer-space` | `SeparateFreezerRack` |
| HARD | `13-clearing-table` | `CandleCleanup` |
| HARD | `14-microwaving-food` | `PlaceMicrowaveSafeItem` |
| HARD | `15-storing-leftovers` | `StoreLeftoversInBowl` |

EASY/MEDIUM primarily use open surfaces; HARD includes doors and enclosed cavities.
This table is maintainer information and is not shipped to the agent. The full
training/held-out split lives in [harness/config.py](harness/config.py).

## Workflow and public API

Development has 8 hours and 75,000 interactions. Each of five evaluation steps has
1 hour and 5,000 actions. Practice episodes use the same 5,000-action ceiling.
Development samples RoboCasa's `pretrain` split; evaluation uses `target` scenes.

The agent sees three RGB/depth cameras and robot proprioception, without object
poses or a provided detector. It builds perception from observations. All actions
pass through the metered client; the simulator is never imported into its process.

```python
from harness.client import SimClient, ObsSpec

with SimClient() as sim:
    tasks = sim.list_tasks()                      # development training set only
    obs = sim.reset(task=tasks[0])["obs"]
    look = sim.observe(ObsSpec(width=512, depth=True))
    print(sim.task_info(), sim.status())
    # Write reusable tools and MANUAL.md under /workspace/agent_harness.
    # Call sim.end_development() when ready, then end the agent turn.
```

`step()` accepts one 12-D action or 1–200 actions in a batch. Observation is free;
development actions and resets cost interactions. Only the current environment is
retained, so switching training tasks rebuilds it. Rendering always uses 512×512;
default delivery is 256×256. RGB/depth are cached per state and resized on request.
Use `ObsSpec(cameras=())` for proprioception alone.

Evaluation starts with its assigned trial already open. `task_info()["instruction"]`
provides the goal; `finish_trial()` gives up the attempt. Neither `reset()` nor
`next_trial()` is available. End the turn after the attempt ends; Harbor opens the
next step. The generated instructions contain the full contract.

### File handoff

Runs do not resume the development conversation. The root hook freezes
`/workspace/agent_harness` once, then restores that same snapshot before every
evaluation. It is on `PYTHONPATH`; dependencies must be regular files inside it.
Symlinks and special files are rejected. Workspace experiments and evaluation edits
do not transfer, and known agent CLI conversation stores are cleared. Detached
agent processes are stopped at step boundaries.

The steps share a container: this is a controlled file handoff, not a fresh-container
sandbox against every possible covert channel. Keep the deliverable directory named
`agent_harness`, so it does not shadow the simulator's `harness` package.

## Scoring

```text
trial_score = (best_count − reset_count) / (total − reset_count)
reward = mean trial_score over all planned evaluation trials
```

[harness/stages.py](harness/stages.py) mirrors each held-out task's success conjuncts.
The scorer retains the best whole stage vector observed at one instant; it does not
combine individually satisfied stages from different times. Conditions already true
at reset earn no free credit. Complete simulator-confirmed success scores 1;
do-nothing and unreached trials score 0. `success_rate` is reported separately.
Development interaction is a hard cap, with no efficiency bonus.

Each verifier writes cumulative reward; `multi_step_reward_strategy = "final"`
selects the last one. Scores come only from private simulator evidence. The original
completion-oriented agent instructions are preserved; this README describes the
implemented partial-credit scorer for maintainers.

## Isolation and failure handling

One root broker grants an exclusive client lease and controls one physics/GL worker.
A second client receives `busy`; disconnecting preserves the scene and idle clients
have no timeout. The root-only SQLite ledger records charges before dispatch and
confirmed evidence afterwards. A worker failure retires the current trial with
`sim_error`, retaining confirmed credit. Subsequent Harbor steps can start new
workers. Lost actions are not replayed, and verification does not need a live simulator.

`/opt/src`, `/opt/private` and `/var/lib/rlebench` are inaccessible to the agent.
The held-out task class, seeds and stage functions stay in private code; the public
interface supplies the task instruction without exposing the hidden class name.
Root workflow control is separate from the public socket. Prior verifier JSON is
removed before the next agent starts.

## Configuration and layout

| Setting | Default / location |
| --- | --- |
| Group | `RLEBENCH_GROUP`, generated slug |
| Development budget | `RLEBENCH_INTERACTION_STEPS=75000` |
| Practice/evaluation horizon | `RLEBENCH_MAX_STEPS_PER_TRIAL=5000` |
| Reported clocks | `RLEBENCH_DEVELOP_SECONDS=28800`, `RLEBENCH_TRIAL_SECONDS=3600` |
| Clock multiplier | `RLEBENCH_TIMEOUT_MULT=1` |
| Number of evaluation steps | `TRIALS_PER_TASK=5` in `harness/config.py`; regenerate after changing |
| Handoff directory | `/workspace/agent_harness` |

Generated `task.toml` forwards budget and multiplier overrides from the host; other
clock overrides must be added to its environment table. Keep reported clocks and
Harbor step timeouts aligned. Pair `RLEBENCH_TIMEOUT_MULT` with Harbor's
`--agent-timeout-multiplier`; the actual deadline is enforced by Harbor, while the
reported phase clock begins at the first API request.

```text
harness/                 private split, stage predicates, task adapter and backend
  client.py              public SimClient/ObsSpec exports
  runtime/               broker, worker, ledger, handoff and verifier
_template/
  image/                 Dockerfile and startup/control/verification/isolation scripts
  steps/{develop,eval_01}/   instruction templates; eval template reused for every trial
build_groups.py          emits selected groups through rlebench/runtime_build.py
build_assets.py          staging/check entry point
dev/                     primitive audit generator and checked-in audit/baseline data
NN-slug/                 GENERATED Harbor tasks with develop + eval_NN steps
image/                   GENERATED shared public/private build payloads
```

To change the split, edit `SPLITS` in `harness/config.py` and check its primitive
coverage against `dev/data/task02_primitive_audit.json`. A new held-out task needs a
matching stage function and validation against RoboCasa's success predicate. Preserve
existing slugs. The current generator supports `--list`, `--cell`, `--emit-all` and
`--check`; `--check` checks payload boundaries, not primitive coverage.

## Build and run

Requires Docker/Compose, an NVIDIA GPU with the Container Toolkit, Harbor and the
complete merged RoboCasa assets. No perception-model download is required.

```bash
make sim-robocasa
make task02-06-setting-the-table             # one representative group
make task02                                 # all groups, one shared image
make task02-assets                          # staging without Docker builds
make task02-clean
```

`rlebench prepare task02` builds the simulator and task. The image is
`rlebench-task02-agent:dev`; build-time `AGENT_UID` defaults to 1000 and cannot
be root.

```bash
rlebench run task02/06-setting-the-table -a <agent> -m <model>
rlebench run task02 -a <agent> -m <model> --device cuda:0 cuda:1
rlebench run task02/06-setting-the-table -a oracle --device cuda:0
```

The runner supplies GPU placement, the dataset mount, agent-host allowlist and Harbor
GPU override; `--dry-run` prints the command. Oracle only checks the six-step protocol
and normally earns zero. It does not establish that a useful transferable harness
has been learned.

## Results and checks

| Path within a Harbor trial | Contents |
| --- | --- |
| `steps/eval_05/verifier/reward.json` | Final cumulative reward, success rate, trial counts and infrastructure failures |
| `steps/eval_NN/verifier/diagnosis.json` | Ledger/handoff status and optional media/error diagnostics |
| `steps/eval_NN/verifier/media/` | Best-effort observation videos and index |
| `steps/develop/artifacts/workspace/agent_harness/` | Development deliverable |
| `steps/*/agent/` | Separate development/evaluation sessions |

Stage evidence remains in the private ledger; diagnosis JSON does not export stage
vectors. Inspect `/var/lib/rlebench/{worker,service,verifier}.log` as root while the
container exists. `rlebench view jobs` shows collected verifier media.

```bash
python3 tasks/task02/build_groups.py --check
python3 tasks/task02/build_assets.py --check
make test TASK=task02
python3 tests/runtime/container_check.py --family task02 \
    --group 06-setting-the-table --gpu 0 \
    --assets third_party/robocasa/robocasa/models/assets \
    --report /tmp/task02-container.json
```

The real-container check exercises UID isolation, rendering, locking and failure
recovery. In a started, idle container, run
`docker exec --user agent <container> python /opt/check_isolation.py`.
Check GPU/assets for startup failures, close the lease owner for `busy`, and inspect
private worker logs for `sim_error`. Keep a vision-capable agent setup: camera access
is part of the task.

Primitive coverage does not establish that three procedures suffice to learn a
transferable harness; that is the benchmark's hypothesis. Step caps and wall clocks
both constrain attempts, and trials run sequentially in the shared container.
