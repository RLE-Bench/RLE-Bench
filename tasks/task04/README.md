# Task 04 — Whole-body motion tracking

Five independent Harbor tasks, one per LAFAN1 clip. The agent writes the whole
training pipeline for a Unitree G1 tracking policy against a frozen deployment
contract, and the verifier re-runs the exported policy in a different simulator
under hidden dynamics randomization.

| task                 | clip                     | frames      |
| -------------------- | ------------------------ | ----------- |
| `01-dance`           | `dance1_subject2`        | `122:722`   |
| `02-fight`           | `fight1_subject2`        | `650:1250`  |
| `03-fall-and-get-up` | `fallAndGetUp2_subject2` | `1:601`     |
| `04-run`             | `run1_subject2`          | `3266:3866` |
| `05-sprint`          | `sprint1_subject2`       | `1525:2125` |

`motions.tsv` defines the matrix. Each clip is a 20 s excerpt of the retargeted
G1 dataset, converted from 30 Hz to 50 Hz by `sim/motiontrack/motiontrack.sh`.
All five tasks share one agent image and one verifier image.

## How it works

The agent gets four hours on one GPU, a batched MuJoCo-Warp environment,
the evaluator, an export helper and a reference PPO recipe at
`/workspace/reference_ppo.py`. The deliverable is an ONNX policy plus metadata
under `/logs/artifacts/policy/`, exported through `harness.export`, with at most
10 M parameters. The observation layout, action mapping and 50 Hz control rate
are fixed in `harness/spec.py`; everything else is the agent's.

The verifier is a separate CPU image. It validates the graph, runs it twice
on a smoke seed to check stability and determinism, then scores eight hidden
seeds in MuJoCo-C at a 2 ms timestep with per-seed friction, mass, centre of
mass, gain, latency, pushes and sensor noise.

### Scoring

```
episode = 0.7 * tracking_multi + 0.3 * survival
        - 0.15 * min(action_jerk / jerk_cap, 1) - 0.05 * fell
reward  = mean over the eval seeds, clipped to [0, 1]
```

`tracking_multi` averages Gaussian kernels of 0.3, 0.1 and 0.05 m over the
re-anchored body position error, with post-fall frames scoring zero. A
submission that fails validation, diverges, or is not bit-reproducible is
capped at 0.1. Per-motion thresholds live in `harness/config.py` and are set by
`python -m dev.calibrate --emit <policy dir> --motion <npz>` from the reference
policy under `dev/oracle_ref/`.

## Layout

```
build_tasks.py           emits NN-slug/ from _template/ and motions.tsv; --check
motions.tsv              the matrix
harness/                 spec, robot, motion, env (Warp trainer), mdp, evaluator
                         (MuJoCo-C), export, progress; verifier-only: scorer,
                         config, eval_seeds.json
dev/                     calibrate.py and the dance reference policy
_template/
  task.toml.in, instruction.md.in
  environment/           agent Dockerfile, compose GPU overlay, dev_eval.py
  solution/              Oracle: solve.sh runs payload/train.py for one hour
  tests/                 verifier Dockerfile, test.sh, score_task.py

NN-slug/                 GENERATED: the five Harbor tasks
```

Both Dockerfiles build from the repo root and copy the harness modules by name;
`tests/test_motiontrack_task.py` checks that the scorer, thresholds and seeds
never enter the agent image. Version pins come from `sim/motiontrack/pins.env`.

## Running it

```bash
make sim-motiontrack   # G1 model, LAFAN1 clips, .venv-motiontrack
make task04            # emit the matrix, convert clips, build both images
make task04-assets     # regenerate only
make task04-clean
```

`rlebench prepare task04` runs both steps.

```bash
rlebench run task04/02-fight -a <agent> -m <model>          # one task
rlebench run task04 -a <agent> -m <model> -d cuda:0 cuda:1  # all five
rlebench run task04/01-dance -a oracle                       # the reference PPO
```

By hand, `RLEBENCH_GPU` picks the card (0 by default); agents other than the
Oracle need the hosts they install from, which `rlebench run --dry-run` prints:

```bash
RLEBENCH_GPU=1 harbor run -p tasks/task04 -i 01-dance -a oracle
```

Set `ORACLE_MINUTES` to a small number to smoke-test the Oracle plumbing.

## Checking the guarantees

```bash
python3 tasks/task04/build_tasks.py --check
make test TASK=task04
make test-task04-golden     # the reference policy still scores at its bar
make calibrate-task04       # re-pin the dance thresholds from the reference
```
