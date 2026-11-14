# Task 06 — Blind multi-shape pose estimation

A robot-mounted stick pushes one of three red, asymmetric blocks (T, C or F)
on a table. From a fixed oblique camera the agent estimates the block's
planar pose and identifies its shape, through rotation and occlusion, with
CPU inference fast enough for real use. The agent never sees the meshes or
the shape drawn for a scene; it learns them from rendered examples.

Four Harbor subtasks share the renderer and verifier and differ in what the
agent gets and may use:

| subtask | input | method | hardware |
| --- | --- | --- | --- |
| [rgb-only](rgb-only/) | RGB | numpy, scipy, OpenCV on CPU | CPU |
| [rgb-depth](rgb-depth/) | RGB-D | numpy, scipy, OpenCV on CPU | CPU |
| [rgb-depth-model-training](rgb-depth-model-training/) | RGB-D | a TorchScript model trained from scratch, at most 20M tensor elements | 1 GPU |
| [method-agnostic](method-agnostic/) | RGB-D | anything, including torch and scikit-learn | 1 GPU |

## How it works

The agent container runs a root-owned rendering service that owns MuJoCo and
the meshes. The workspace holds a client for labeled design renders on the
public seeds (pose labels are noisy and carry no shape label) and a
diagnostic evaluator limited to six calls. The submission is
`/logs/artifacts/estimator.py` exposing `make_estimator()`, or `model.pt`
for the model-training subtask, collected within a two-hour session.

The verifier is a separate CPU image that renders its own battery on
unpublished seeds: 100 independent Stage A frames and ten Stage B push
episodes of 60 frames with the arm crossing the line of sight. It masks the
observation to the subtask's modalities, runs the submission in a sandbox as
an unprivileged user with the scoring code locked away, and grades only the
poses and shape IDs that come back.

### Scoring

```
reward = 0.3 * stage_a + 0.7 * stage_b - efficiency deduction
```

Each Stage A group of ten frames and each Stage B episode scores the mean of
two Gaussian kernels over its five largest translation and rotation errors,
and scores zero if any shape ID in the group is wrong. The only gate is that
the submission loads and returns one finite pose; failing it scores zero.
Inference slower than 10 Hz on the verifier's CPU loses up to 0.2. Kernel
widths and weights live in `harness/config.py`; the evaluation seeds in
`harness/eval_seeds.json` are verifier-only.

## Layout

```
build_assets.py          generates the four subtask directories, their agent,
                         private, verifier and Oracle trees
harness/
  spec.py, public_spec.py, config.py, eval_seeds.json
  scene.py, sensor.py, episodes.py, battery.py       scene, sensor model, push episodes
  training_service.py, training_client.py            root-owned renderer and its client
  design_evaluator.py, evaluation_client.py          agent-side diagnostic
  sandbox.py, checkpoints.py, scorer.py, render.py   verifier
  oracle_cv.py, oracle_geom.py, oracle_fused.py, oracle_unknown.py, oracle_cnn.py, baselines.py
  torch_contract.py, train_harness.py                model-training contract and utilities
  assets/tshape/         the three block meshes and the pusher stick
dev/                     calibrate.py (battery audit), audit_motion.py, preview_motion.py
<subtask>/               task.toml, instruction.md, README.md, environment/, tests/, solution/

<subtask>/environment/{agent,private}, tests/{harness,assets}, solution/payload   GENERATED
```

Everything under the subtask directories is written by `build_assets.py`;
edit the templates and the harness, then regenerate.

## Running it

```bash
make task06            # stage the trees and build every subtask image
make task06-assets     # regenerate only
make task06-clean
```

```bash
rlebench run rgb-only -a <agent> -m <model>
rlebench run task06/method-agnostic -a oracle
```

By hand:

```bash
harbor run -p tasks/task06/rgb-only -a oracle
RLEBENCH_GPU=1 harbor run -p tasks/task06/rgb-depth-model-training -a oracle  # card 0 by default
```

`verifier/report.json` carries the per-group scores and medians;
`verifier/media/` holds one video per Stage B episode with truth and
estimate drawn.

## Checking the guarantees

```bash
make test TASK=task06
make test-task06-metrics    # scene, sensor and reference-estimator checks
make test-task06-golden     # scoring mechanics and the anti-gaming battery (heavy)
make calibrate-task06       # audit the battery and print reference and probe scores
```
