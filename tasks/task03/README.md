# Task 03 — Tabletop reasoning

Five tabletop tasks solved from RGB images and robot proprioception through a
small public client. The simulator, scene definitions, hidden parameters and
scoring records are root-only; the agent acts through a metered Unix socket.

| Task | Goal | Budget | Image |
| --- | --- | --- | --- |
| `01-tower-max-height` | Build a tall stable tower | 50,000 steps | `rlebench-task03-agent:dev` |
| `02-cantilever-overhang` | Extend a stack beyond the table edge | 50,000 steps | `rlebench-task03-agent:dev` |
| `03-balance-coins` | Identify the heavy cube using a balance | 50,000 steps | `rlebench-task03-agent:dev` |
| `04-rubik-cube` | Physically solve a 2×2 cube with two arms | 50,000 steps | `rlebench-task03-pocket-agent:dev` |
| `05-hidden-center-of-mass` | Identify ballast quadrants in three sealed boxes | 12,000 steps per box | `rlebench-task03-hidden-com-agent:dev` |

## Workflow and public APIs

There is no development/evaluation split. Tasks 01–04 each have one nine-hour
attempt; HiddenCOM has one hour for all three boxes. Public API and scene geometry
are documented in [harness/client.py](harness/client.py), installed at
`/opt/rlebench/harness/client.py` and accessible through Python `help()`.

### Tower, cantilever and balance

`TabletopClient` exposes `observe`, `step`, `move`, `reset` and `finish` for a
PandaOmron arm, gripper, mobile base and torso. Control runs at 20 Hz with 12-D
actions; `move` tracks a world-frame tool pose through the same metered controller.

```python
from harness.client import TabletopClient
from PIL import Image

with TabletopClient() as sim:
    obs = sim.observe()
    Image.fromarray(obs["images"]["left"]).save("/workspace/left.png")
    # Manipulate with step()/move(), then finish() to submit the final scene.
```

`reset()` costs one step and restores the same private initial scene. `finish()`,
budget exhaustion or the end of the Harbor attempt submits the final simulator
state. Release the structure before finishing. No depth, masses, contact readings
or general object poses are supplied. BalanceCoins additionally exposes cube and
pan centres; the heavy cube's identity stays private.

### Physical cube

`SpeedrunClient` controls two tilted Panda arms through 14-D actions and a front RGB
camera with optional depth. `task_info()` describes the robot frames and camera.
There is one trial: `reset()` forfeits it, and physical success ends it automatically.
`recover_drop()` restores a dropped cube to its support for one metered step without
rerolling the scramble or refilling the allowance. The public helper and manual are
staged from `_template/cube_recovery.py` and `_template/pocket_manual.md`.

### Hidden center of mass

`HiddenCOMClient` controls a fixed Panda with 7-D actions and three RGB cameras.
Submit one immutable A/B/C/D answer per box with `submit()`. The first two submissions
advance to the next box; observe before acting again. The final reply has `done=True`.
There is no reset or correctness feedback. At the movement cap the agent can still
observe and submit. After `sim_error`, a valid submission advances without credit
for that failed box.

All variants grant one connected client exclusive control. A second client receives
`busy`; disconnecting releases the lease without resetting the scene. Observation
is free, batches contain 1–200 actions, and requests exceeding the remaining budget
are rejected. All cameras render at 512×512; supported smaller image requests resize
those frames. Tabletop/HiddenCOM convenience observations use 512×512 RGB.

## Scoring

The verifier uses committed simulator evidence, never an agent's written answer or
reported metric. Current reward is normalized task quality, with no additional
control-step discount:

| Task | Quality / reward |
| --- | --- |
| Tower | Final stable height relative to the analytic optimum |
| Cantilever | Final stable overhang relative to the analytic optimum |
| BalanceCoins | Correct final selection, discounted for excess weighings |
| Cube | Zero unless physically solved; solved quality decays with excess quarter turns |
| HiddenCOM | Fraction of the three submitted quadrants that are correct |

For a nontrivial cube scramble, solved quality is
`2 ** -max(0, actual_qtm / optimal_qtm - 1)`: optimal scores 1, twice optimal 0.5.
Unclassified transitions cap quality at 0.5. The action budget still applies.

The original agent wording is preserved, including the cube's binary-success
sentence and HiddenCOM's request to minimize steps. This maintainer description
reflects the implemented scorer. The old README's
`max(0, quality - 0.5 * steps_used / step_budget)` formula is not used by this runtime.

## Isolation, recovery and layout

The family owns its [harness/](harness/) adapters and shares the
[broker/worker runtime](../../rlebench/runtime/) with task01/02. Domain physics in [tabletop/](tabletop/) is staged privately
as `harness.tabletop`; it is not assembled from task01 at build time.

The root broker records charges before dispatch and confirmed evidence afterwards
in SQLite. One worker owns physics and GL, with supervision and bounded operation
timeouts. A simulator failure closes the affected attempt; confirmed evidence
remains available to the independent verifier. HiddenCOM can advance to remaining
boxes. Optional media encoding cannot prevent numeric reward generation.

`/opt/src`, `/opt/private` and `/var/lib/rlebench` are root-owned and inaccessible to
the agent. Only public client/protocol code, documentation and variant-specific
helpers enter `/opt/rlebench`. Hidden masses, seeds and scoring stay private.

```text
harness/                 tracked client, task configuration and adapter
../../rlebench/runtime/  shared simulator service and public client
tabletop/                private scenes, analytic models, budgets, pocket/, hidden_com/
_template/
  instruction.md.in, tasks.json   tower/cantilever/balance instructions
  pocket.md, pocket_manual.md, cube_recovery.py
  hidden_com.md, hidden_com_solution/
  image/                 Dockerfile and runtime/isolation scripts
  solution/              recorded tabletop replay Oracles
build_levels.py          emits any of the five tasks through rlebench/runtime_build.py
build_assets.py          staging/check entry point
dev/                     host-only references and analysis
NN-slug/                 GENERATED Harbor tasks
image/                   GENERATED tabletop image context
image-pocket/            GENERATED cube image context
image-hidden-com/        GENERATED HiddenCOM image context
```

Edit the tracked sources and templates, then regenerate. Budgets live in
`tabletop/budgets.py`; task schedules in `harness/task.py`; generated Harbor timeouts
in `rlebench/runtime_build.py`. Keep instructions and timeouts aligned when changing
budgets. `AGENT_UID` defaults to 1000 and cannot be root.

## Build and run

Requires the pinned RoboCasa simulator base image, Docker/Compose, an NVIDIA GPU
with the Container Toolkit and Harbor. Task03 does not mount the RoboCasa dataset
or require perception models. MuJoCo and robosuite come from the shared base image.

```bash
make task03 TASK_CELLS=01-tower-max-height    # one representative task/image
make task03                                 # all five tasks, three images
make task03-assets                          # staging without Docker builds
make task03-clean
```

`rlebench prepare task03` runs the task build; the shared simulator image must already
be available (see [sim/robocasa](../../sim/robocasa/README.md)).

```bash
rlebench run task03/01-tower-max-height -a <agent> -m <model> --device cuda:0
rlebench run task03 -a <agent> -m <model> --device cuda:0
rlebench run task03/01-tower-max-height -a oracle --device cuda:0
rlebench run task03/05-hidden-center-of-mass -a oracle --device cuda:0
```

The runner handles GPU placement and agent-host access; `--dry-run` prints its
Harbor command. Tabletop Oracles replay recorded actions through the public API;
HiddenCOM's Oracle estimates tilt from RGB. A completed replay is not a guarantee
of an optimal score. The cube has no Oracle, so select supported cells explicitly.

## Results and checks

For each Harbor trial, `verifier/reward.json` contains reward, trial counts and
infrastructure failures. Private family metrics add the legacy quality, budget,
cube-turn and per-box breakdowns. Tabletop `success_rate` retains its continuous
quality meaning; `quality` names it explicitly. Cube quality and `task_score`
are reported before the unclassified-turn reward cap.
`verifier/diagnosis.json` reports ledger/handoff status.
`verifier/media/` contains best-effort observation videos and an index. Agent sessions
are under `agent/`, and collected files under `artifacts/workspace/`.
`rlebench view jobs` displays results and media. Private logs are under
`/var/lib/rlebench` while the container exists, and exported to `verifier/diagnostics/`
after the final agent stops. Failure entries in diagnosis JSON identify the operation
and failure category without exposing private simulator state.

```bash
python3 tasks/task03/build_levels.py --check
python3 tasks/task03/build_assets.py --check
make test TASK=task03
python3 tests/runtime/container_check.py --family task03 --task TowerMaxHeight \
    --gpu 0 --report /tmp/task03-tabletop.json
python3 tests/runtime/container_check.py --family task03 --task RubikCube \
    --variant pocket --gpu 0 --report /tmp/task03-cube.json
python3 tests/runtime/container_check.py --family task03 --task HiddenCOM \
    --variant hidden-com --gpu 0 --report /tmp/task03-hidden-com.json
```

Build each image before its container check. These checks exercise real rendering,
controls and exclusivity, and verify as the agent UID that private files, modules
and the control socket are inaccessible. In a started, idle container, use
`docker exec --user agent <container> python /opt/check_isolation.py`.

For `busy`, close the current lease owner. For `sim_error` or startup failures,
inspect container/worker logs and GPU access. A single-attempt task cannot restart
a failed graded scene; the verifier still reports its committed score and diagnostics.
