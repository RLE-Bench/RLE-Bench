# Task 01 — RoboCasa speed-run

The agent learns a kitchen manipulation task through a metered simulator, then
passes one terminal evaluation. The score measures **capability first, sample
efficiency second**: how many trials it solves and how much interaction it spent
learning. Three harness levels × five subtasks produce fifteen Harbor tasks.

| Level | What the agent receives | Image |
| --- | --- | --- |
| L1 | Three RGB/depth cameras, robot proprioception, low-level actions | `rlebench-task01-l1-agent:dev` |
| L2 | L1 plus client skills, perception services and `/opt/HARNESS_MANUAL.md` | `rlebench-task01-l2-agent:dev` |
| L3 | L2 plus selected object/fixture poses, extents and grasp state under `priv_*` keys | `rlebench-task01-l3-agent:dev` |

Subtasks: `01-open-fridge`, `02-close-cabinet`, `03-turn-on-stove`,
`04-pick-place-counter-to-drawer`, `05-pick-place-microwave-to-counter`.
The task, evaluation plan, seeds and reward weights are the same across levels;
only the observation and tool access changes.

## Workflow and public API

| Harbor step | Default allowance | Purpose |
| --- | --- | --- |
| `develop` | 8 hours; 100,000 interactions | Practise, inspect observations and write code |
| `evaluate` | 1 hour; five trials of at most 1,000 actions each | Apply the developed policy to hidden kitchens |

Both steps share `/workspace`. `rlebench run` requests trajectory resumption for
task01 agents, except Oracle; the selected agent must support it. The root Harbor
hook opens evaluation after development ends. `end_development()` closes agent
interaction in development; it does not itself start evaluation.

```python
from harness.client import SimClient, ObsSpec

with SimClient() as sim:
    info = sim.task_info()                  # action layout, cameras and limits
    obs = sim.reset()["obs"]               # development only; costs one interaction
    look = sim.observe(ObsSpec(width=384, depth=True))
    print(sim.status())
    # Develop a controller using sim.step(action_or_batch, obs_spec).
    # Call sim.end_development() when ready, then end the agent turn.
```

In evaluation, `finish_trial()` ends the current attempt and `next_trial()` opens
its successor; call it only after the current episode ends. `reset()` is unavailable.
End the agent turn once the phase is `finished`. Full examples are in the generated
step instructions and [public client](harness/runtime/client.py).

Actions have 12 components in the robot's base frame. Batches contain 1–200 actions
and cannot exceed the remaining allowance. `observe()` is free; development resets
and actions are charged. Evaluation actions have their own trial cap and do not
reduce the development-efficiency score. Default images are 128×128. Rendering is
always 512×512, with RGB and metric z-depth cached per state and resized for delivery.
`ObsSpec(cameras=())` omits images; depth is optional.

## Scoring

```text
reward = success_rate × (0.80 + 0.20 × (1 − development_steps / interaction_budget))
```

Success comes from the simulator's predicate. Efficiency earns nothing without
success. The default evaluation plan has five trials in five distinct kitchens;
unreached trials count as failures. The verifier reads committed private evidence,
never an agent-written score or an exported copy of the ledger.

## Simulator isolation and recovery

The public Unix socket grants control to one connected client. Another client gets
`RemoteError(kind="busy")`; closing a connection releases control without resetting
the scene. Idle connections have no timeout. The agent has no in-process simulator.

A root broker owns the ledger and supervises one physics/GL worker. Actions are
charged before dispatch, confirmed evidence is committed separately, and failed
requests are never blindly replayed. A worker failure ends the attempt with
`sim_error`; development can reset and evaluation can advance to the next trial.
The launcher restarts the broker, and root handoff hooks are idempotent. Verification
reads SQLite independently of simulator availability; media failure does not remove
the numeric reward.

`/opt/src`, `/opt/private` and `/var/lib/rlebench` are root-owned and inaccessible to
the agent. The public payload contains only the client/protocol and level-appropriate
skills; simulator code, hidden plans, counters and scoring stay private. L3 exposes
only the deliberate privileged observation fields.

## Configuration and layout

| Setting | Default / location |
| --- | --- |
| Task and level | `RLEBENCH_TASK`, `RLEBENCH_LEVEL`, generated per cell |
| Development budget | `RLEBENCH_INTERACTION_STEPS=100000` |
| Trial horizon | `RLEBENCH_MAX_STEPS_PER_TRIAL=1000` |
| Reported clocks | `RLEBENCH_DEVELOP_SECONDS=28800`, `RLEBENCH_EVALUATE_SECONDS=3600` |
| Clock multiplier | `RLEBENCH_TIMEOUT_MULT=1` |
| Reward weights | `RLEBENCH_W_OUTCOME=0.8`, `RLEBENCH_W_EFFICIENCY=0.2`, verifier environment only |
| Hidden plan / seed salt | Root-only `/opt/private/eval_plan.txt`; `plan=1x1,2x1,3x1,4x1,5x1` and optional `salt=...` |

The generated `task.toml` forwards budget and multiplier overrides from the host.
Other overrides require editing the appropriate environment table. Never place
hidden seeds or plans in container-wide environment variables. Reported time starts
at the phase's first API request; Harbor enforces the actual deadline. When scaling
it, pair `RLEBENCH_TIMEOUT_MULT` with Harbor's `--agent-timeout-multiplier`.

```text
harness/                 task.py/config.py, adapter/backend, public client and skills
  runtime/               broker, worker, ledger, control hooks and verifier
_template/
  image/                 Dockerfile, startup/control/verification/isolation scripts
  steps/{develop,evaluate}/   instructions and level-specific harness descriptions
build_levels.py          family entry point for the host generator
build_assets.py          staging/check entry point
images/L1|L2|L3/          GENERATED public/private image payloads
L1|L2|L3/NN-slug/         GENERATED Harbor cells, steps, tests and protocol Oracles
```

The host generator is [rlebench/runtime_build.py](../../rlebench/runtime_build.py).
Generated trees are gitignored. Edit sources and templates, then regenerate.

## Build and run

Requires Docker/Compose, an NVIDIA GPU with the Container Toolkit, Harbor, and the
merged RoboCasa assets. L2/L3 also require the pinned perception sources and weights.

```bash
make sim-robocasa
make task01-L1 TASK_CELLS=01-open-fridge       # small development build
HF_TOKEN=... make sim-perception             # gated SAM3 weights, needed for L2/L3
make task01                                 # all fifteen cells and three images
make task01-assets                          # stage all cells without Docker builds
make task01-clean
```

`TASK01_LEVELS=L1 make task01` builds just L1. `AGENT_UID` defaults to 1000 and must
not be root. `rlebench prepare task01` builds the simulator and task images; prepare
the perception dependencies separately before requesting L2/L3.

```bash
rlebench run task01/L1/01-open-fridge -a <agent> -m <model>
rlebench run task01/L2 -a <agent> -m <model> --device cuda:0 cuda:1
rlebench run task01/L1/01-open-fridge -a oracle --device cuda:0
```

The runner supplies the dataset mount, GPU placement, agent-host allowlist and
Harbor GPU override. `--dry-run` shows the generated command. Oracle only exercises
the protocol and normally scores zero; it is not a task solution.

## Results and checks

Under a Harbor trial directory:

| Path | Contents |
| --- | --- |
| `steps/evaluate/verifier/reward.json` | Reward, success rate, planned/recorded/attempted trials, development steps and infrastructure failures |
| `steps/evaluate/verifier/diagnosis.json` | Ledger and handoff status; media/error diagnostics when applicable |
| `steps/evaluate/verifier/media/` | Best-effort observation videos and `index.json` |
| `steps/*/agent/` | Agent sessions |
| `steps/*/artifacts/workspace/` | Collected workspace files |

Only the final step reward counts. Check infrastructure failures and diagnosis before
interpreting a low score. Root-only `worker.log`, `service.log` and `verifier.log` live
in `/var/lib/rlebench` while the container exists. Videos sample observations rather
than every physics step; `rlebench view jobs` displays collected verifier media.

```bash
python3 tasks/task01/build_levels.py --check
python3 tasks/task01/build_assets.py --check
make test TASK=task01
python3 tests/runtime/container_check.py --family task01 --task OpenFridge \
    --level L1 --gpu 0 --assets third_party/robocasa/robocasa/models/assets \
    --report /tmp/task01-container.json
```

The generator checks staged public/private boundaries. The container check exercises
real reset/step/render, exclusive control and UID isolation; repeat for L2 and L3
after building them. In an already started, idle container, run
`docker exec --user agent <container> python /opt/check_isolation.py`.

| Symptom | Check |
| --- | --- |
| Missing `ROBOCASA_ASSET_DIR` | Use `rlebench run` or supply the complete asset path |
| Image/level mismatch | Rebuild the requested level |
| `busy` | Close the client holding the simulator lease |
| `sim_error` | Inspect private worker logs; reset in development or advance evaluation |
| Startup health failure | Inspect container logs, assets and GPU access |
| Zero attempted trials | Inspect development timeout and evaluation agent logs |

L1/L2 depend on visual observations; use an agent setup that can inspect images.
