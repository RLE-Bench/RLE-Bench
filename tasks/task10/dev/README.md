# Difficulty validation

This directory contains pre-harness calibration experiments for the proposed Fault-Adaptive Arm Control task. These experiments are not agent-facing and do not establish benchmark difficulty by themselves.

## Phase 1: fault sensitivity

`difficulty_probe.py` runs a torque-controlled Panda model with deterministic draws of encoder zero offset, actuator gain error, viscous friction, command delay, a weakened joint, and payload scaling. It compares a nominal inverse-dynamics controller, a fixed-gain conservative controller, and a privileged oracle ceiling.

The 24-seed v1 run produced the following mean scores: nominal 0.635, conservative 0.573, and oracle 0.858. The corresponding mean joint RMSE values were 0.113, 0.118, and 0.040 radians. No controller crossed the provisional hard joint limits. Repeating the complete run produced byte-identical JSON.

The delay bands expose a useful gradient. For zero, one, and two control-step delays, nominal scored 0.657, 0.652, and 0.593; the oracle scored 1.000, 0.869, and 0.671. However, this experiment does not yet meet the acceptance bar: the fixed-gain conservative controller underperforms the nominal controller, some nominal instances score above 0.9, and the oracle minimum is only 0.562.

## Decision

Continue validation, but do not label the task calibrated or PR-ready. Phase 2 must introduce a non-privileged online identification baseline, separate calibration from evaluation, add the payload-contact and mid-run derating scenarios, and score by the worst public fault band. The task is a credible hard-task candidate only if those changes produce a stable ordering from nominal to robust to adaptive to golden without allowing the nominal controller to saturate the benchmark.

## Phase 2: observation-only identification

The second probe adds a 120-step calibration window and an adaptive baseline that estimates encoder offset from the known reset pose and actuator gain and command delay from commanded torque and measured motor current. It does not read the sampled fault parameters. Across 24 seeds, the fixed nominal controller scored 0.634, the conservative controller 0.552, the adaptive controller 0.821, and the privileged oracle 0.820. Adaptive and oracle are statistically tied at this probe's resolution because both retain an unmodeled failure on seed 3 and the score includes control effort.

This establishes that the public observation contract contains enough information to recover most of the fault penalty, but it still does not prove hard-task quality. The conservative baseline needs a meaningful safety scenario, seed 3 needs diagnosis, and the probe needs Cartesian tracking, contact, impulse recovery, and an online actuator derating before agent pilots begin.

Run the probe with:

```bash
.venv-task10/bin/python tasks/task10/dev/difficulty_probe.py --seeds 24 --json /tmp/task10-difficulty.json
```


## Phase 3: coupled hidden faults and a fixed acceptance gate

The third probe adds adjacent-joint actuator cross-talk, quantized noisy motor-current observations, zero-to-three-step delay, a bounded impulse, and a second weakening of one actuator halfway through evaluation. The observation-only adaptive baseline retains a deliberately incomplete diagonal model, while the privileged oracle uses the true coupled actuator map and forward-simulates already queued commands. Fixing the oracle predictor was necessary to distinguish a genuinely infeasible fault distribution from a weak reference controller.

Before the 24-seed run, the provisional hard-task gate was fixed as follows: nominal mean at most 0.45; adaptive mean from 0.45 through 0.72; oracle mean at least 0.90; oracle minimum at least 0.85; oracle-minus-adaptive mean gap at least 0.25; and zero oracle safety violations. The run passed every condition. Mean scores were 0.423 for nominal, 0.364 for fixed-gain robust, 0.647 for observation-only adaptive, and 0.987 for oracle. The oracle minimum was 0.910, the identification gap was 0.340, and no controller crossed the provisional hard joint limits.

Two complete 24-seed executions produced byte-identical JSON. Both files have SHA-256 7cb043bf30b77d8966db350d623b1e649eb2c9e6edfaecdec999003b06c4da6a.

This validates the difficulty and solvability of the dynamic identification core. It does not yet validate the proposed Cartesian/contact scenarios, the final separate verifier packaging, leakage resistance, or performance of an unprivileged golden controller.

Run the enforced gate with:

    .venv-task10/bin/python tasks/task10/dev/difficulty_probe.py --seeds 24 --check --json /tmp/task10-difficulty.json


## Full Harbor harness validation

The implemented harness adds Cartesian free-space tracking, compliant contact with verifier-measured filtered force, impulse recovery, mid-run actuator derating, a public practice runner, separate verifier packaging, an isolated controller subprocess and an observation-only golden controller. The policy RNG seed is the fixed public value zero and has no reversible relationship to hidden scenario seeds.

Across eight hidden seeds and three scenarios, the final no-network verifier-container Oracle reward is 0.757557: free-space 0.910497, contact 0.543991 and recovery 0.811846. All 24 episodes complete without a safety violation or controller load failure. The nominal controller is capped at 0.100 after a contact safety failure, and the zero controller scores 0.000.

Two complete verifier-container executions produced byte-identical report and reward files. The report SHA-256 is bb9772c8656e60d19d2168a153950806d9b72e93a9275170b0ff0cfee959e882 and the reward SHA-256 is 2e4c4587869ba5bdbb85ff7e9ac7f7e6083bc5eb4767c2d826b8a50cad3b158d.

The root-equivalent isolation test verifies that the worker becomes uid 65534, cannot read verifier files or parent memory, cannot import the verifier harness, cannot fork and cannot affect parent simulation state through its local MJB. Malformed actions, exceptions and hangs are rejected. Host regression tests and Docker dependency-pin tests pass.

This proves deterministic separation between degenerate, nominal and reference methods. It does not establish a frontier-model pass-rate distribution; that requires paid model pilots through Harbor after review.
