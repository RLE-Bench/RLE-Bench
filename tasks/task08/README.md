# Task 08 — One mobile base for three arms

The agent designs a mecanum mobile-manipulator base from stock components and
submits it with a shelf-entry controller. The verifier composes the same base
with canonical Panda, UR5e and xArm7 arms, drives it through a hidden dynamic
battery, and runs the submitted controller against twelve public shelf
targets under payload. Every checkpoint takes the worst arm.

## How it works

The agent's workspace holds the public shelf scene, `shelf_spec.json`, the
three canonical arm models and the stock components (mecanum wheels, battery,
adapter plate, aluminium profiles). It writes `robot.xml` from scratch and
`controller.py`, both collected from `/logs/artifacts/`, within a two-hour
session.

The verifier is a separate CPU image. For each arm it removes the submitted
`arm_*` subtree, inserts the trusted adapter at `arm_mount_site`, attaches the
canonical arm, and then:

- checks validity: the model loads, inertias are realizable, the structure is
  connected through every wheel centre, the base settles and drives
  holonomically;
- measures the design: footprint, chassis mass, recognized profile length,
  battery mount, reach preserved and reach beyond the footprint;
- runs scenarios A–E (static sweep, accelerate/brake, cornering, ramp,
  terrain, combined) on three hidden seeds with its own base controller;
- runs `controller.py` in a uid-isolated process for all twelve targets at
  1 kg and again at 2 kg, replaying the first target to check determinism.

### Scoring

```
reward = sum of weighted checkpoints, each the minimum over the three arms
```

| stage       | weight | checkpoints                                                    |
| ----------- | ------ | -------------------------------------------------------------- |
| validity    | 0.15   | loads*, inertia, structure*, mecanum, equilibrium              |
| design      | 0.35   | still valid, resource targets, design efficiency (0.22), reach preserved, reach beyond, battery |
| static      | 0.15   | static margin over the scenario-A sweep                        |
| dynamic     | 0.20   | scenarios B, C, D1, D2, E                                      |
| integration | 0.15   | shelf targets at 1 kg (0.10) and at 2 kg (0.05)                |

`*` gates: failing one caps the total at 0.15. Design efficiency rewards
remaining mass, footprint-area and profile-length budget with no plateau, so
a lighter, smaller base always scores higher. Fixed settings live in
`harness/config.py`; the stability thresholds in `harness/thresholds.py` are
emitted by `python -m dev.calibrate --write` from the reference design.

## Layout

```
build_assets.py          stages environment/assets, tests/harness, tests/models,
                         solution/payload; builds reference/ on a fresh clone
harness/
  config.py, thresholds.py
  metrics/               FASM, SSM, SSF, ZMP, support polygon
  sim/                   scene composition, scenarios, mecanum kinematics, canonical arms
  base_design/           checkpoints, scorer, shelf scene, controlled pick,
                         controller isolation, renders
  assets/                public scene + shelf spec, stock components, golden robot
dev/                     calibrate.py, build_profile_reference.py, controller isolation check
instruction.md, task.toml, environment/Dockerfile, tests/{Dockerfile,test.sh,score_task.py}
solution/                Oracle: reference robot.xml + standalone shelf controller

environment/assets, tests/harness, tests/models, solution/payload, reference/   GENERATED
```

The Oracle controller is assembled by `build_assets.py` from
`solution/controller.py` plus the harness IK helpers, so it imports nothing
private. The reference chassis under `reference/` is derived from the golden
robot by `dev/build_profile_reference.py`.

## Running it

```bash
make task08            # stage the trees and build both images
make task08-assets     # regenerate only
make task08-clean
```

```bash
rlebench run task08 -a <agent> -m <model>
rlebench run task08 -a oracle            # reference design + controller
```

By hand, the Oracle needs no network; other agents need the hosts they install
from, which `rlebench run --dry-run` prints:

```bash
harbor run -p tasks/task08 -a oracle
```

`verifier/report.json` carries the per-arm reports, the limiting arm and the
per-target shelf results; `verifier/media/` holds renders of the submitted
robot.

## Checking the guarantees

```bash
make test TASK=task08
make test-task08-golden     # the reference design scores at its bar (heavy)
make calibrate-task08       # validate the installed thresholds against the reference
docker run --rm -v "$PWD/tasks/task08/dev:/dev-src:ro" -e PYTHONPATH=/tests \
    rlebench-task08-verifier:dev python3 /dev-src/check_controller_isolation.py
```

The golden suite runs the full battery on the reference design without a
shelf controller; shelf control is validated end to end with the Oracle run.
The isolation check asserts that a submitted controller runs as `nobody`, sees
no verifier files, cannot fork, and cannot alter the simulation it is scored
in.
