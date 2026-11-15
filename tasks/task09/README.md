# Task 09 — GELLO co-design for three followers

The agent receives CAD-assembled GELLO lead arms for the Franka Panda, UR5e
and xArm7 with the correct kinematics but no gravity compensation. For each
it designs the passive hardware (springs, masses, counterweights) in
`lead.xml` and writes `trim.py`, a model-based feedforward that also adapts
to an unknown physical copy from probe measurements. The three variants are
scored independently and the reward is their mean.

## How it works

Each variant has its own workspace under `/workspace/<variant>` with the bare
`lead.xml`, `scene.xml`, the meshes and a public `harness/` with the nominal
model and analysis helpers. Deliverables are collected from
`/logs/artifacts/<variant>/` within a three-hour session.

The verifier is a separate CPU image that scores every variant in its own
interpreter. It treats `lead.xml` and `trim.py` as untrusted input: the model
is recomposed with the pinned stock hardware, all physics is recomputed from
the verifier's own instrumentation, and `trim.py` runs only in an isolated
subprocess, sampled on fixed query matrices and applied as a frozen
feedforward. Per variant it:

- checks validity: the model loads, printed parts are printable, the measured
  chain and stock motors are intact, the arm settles, the mass budget holds,
  springs are physical and the device is buildable;
- runs the hardware battery with the feedforward off on hidden device
  instances (mass jitter plus a handle payload);
- runs the submitted software on a verifier-owned reference arm, nominal and
  adapted from the probe protocol;
- runs hardware and software together: hold, backdrive, servo headroom and
  push recovery.

### Scoring

| stage    | weight | checkpoints                                                |
| -------- | ------ | ---------------------------------------------------------- |
| validity | 0.10   | motion clearance over three trajectories |
| hardware | 0.30   | passive residual, passive hold, passive backdrive |
| software | 0.20   | adapted hold, adapted backdrive |
| codesign | 0.40   | combined hold, combined backdrive, headroom, push recovery |

Each of the ten reward checkpoints has weight 0.10. Basic validity (load,
printability, structure, settle, budget, springs, buildable) awards no credit:
its shortfalls deduct up to 0.10, preserving the checks' relative weights.
The validity stage reports motion-clearance credit minus these deductions;
the final reward is floored at zero. Nominal feedforward remains available
as a fallback and is recorded diagnostically, but no longer earns S1 credit.

Loading or structure failures make the submitted-arm performance metrics
unmeasurable, so those metrics receive zero. There is no additional total-score cap.
Physical checkpoints score
continuously against per-follower bars derived from the reference design,
and a better design keeps earning until each checkpoint saturates. The bars
live in `harness/codesign_variant_config.py`. H1 reaches full credit at its
residual bar; other performance checkpoints keep their existing threefold
headroom. The physical bars are emitted by
`RLEBENCH_GELLO_VARIANT=<variant> make calibrate-task09`.

## Layout

```
build_assets.py          stages environment/assets, tests/harness, tests/models,
                         solution/payload; regenerates the lead models first
harness/
  spec.py, variants.py, codesign_variant_config.py, lead_geometry.json
  scenarios.py, balance.py, torques.py, kinematics.py, validity.py
  codesign.py, codesign_checkpoints.py, codesign_scorer.py, sandbox.py
  codesign_oracle.py     reference lead.xml + trim.py
  assets/gello_codesign/ bare, reference and oracle models per variant
dev/                     assemble.py, generate_variant_assets.py, calibrate.py,
                         ablations/, data/ (pinned reference recipe, calibration records)
instruction.md, task.toml, environment/Dockerfile, tests/{Dockerfile,test.sh,score_task.py}
solution/solve.sh        Oracle: copies the three reference payloads

environment/assets, tests/harness, tests/models, solution/payload, harness/assets/*/meshes   GENERATED
```

The lead models are generated from the measured CAD parts under
`assets/robots/gello_mechanical` by `dev/generate_variant_assets.py`; the
verifier reference arm is pinned in `dev/data/reference_design.json`.
The oracle submission is pinned separately in `dev/data/oracle/<variant>/`
and staged into `lead_oracle.xml`, `trim_oracle.py`, and `solution/payload`.
Its reward is approximately 0.7154 (Franka 0.7580, UR5e 0.7045, xArm7 0.6838)
with equal metric weights, H1's full-credit residual bar, and all validity
and motion-clearance checks passing. This
Astra-derived oracle was optimized with verifier access and uses a known
20/48/76 g payload-class prior, inferred from 16 measurements.

## Running it

```bash
make task09            # stage the trees and build both images
make task09-assets     # regenerate only
make task09-clean
```

```bash
rlebench run task09 -a <agent> -m <model>
rlebench run task09 -a oracle            # reference hardware + software
```

By hand, the Oracle needs no network; other agents need the hosts they install
from, which `rlebench run --dry-run` prints:

```bash
harbor run -p tasks/task09 -a oracle
```

`verifier/report.json` carries the per-variant reports; `verifier/media/`
holds renders of each submitted device at home and after the hardware-only
hold.

## Checking the guarantees

```bash
make test TASK=task09
make test-task09-metrics    # analytic metrics, sandbox plumbing, scoring
make test-task09-golden     # per-variant reference scores and ablation tiers
```
