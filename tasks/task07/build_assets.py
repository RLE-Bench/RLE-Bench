#!/usr/bin/env python3
"""Assemble the task07 Harbor task directory (bin clearing).

Generates task.toml, the agent environment (Dockerfile + public harness
subset + Panda assets + dev runner + instruction.md), the solution payload
(the depth-baseline policy), and the verifier context (full harness + eval
seeds + score_task.py + test.sh). Everything generated is gitignored except
task.toml, the Dockerfiles, test.sh and solve.sh; run this builder (or
``make task07-assets``) after changing tasks/task07/harness/.

A forbidden-token scan fails the build if any agent-shipped file references
the privileged picker, the scorer, the calibration, the oracle payload, or
the evaluation seeds.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "tasks" / "task07"
SRC = REPO / "tasks" / "task07" / "harness"
CORE_FILES = ("__init__.py", "media.py")   # rlebench.core, as runtime.py records with

AGENT_FILES = ("__init__.py", "spec.py", "parts.py", "scene.py",
               "sensor.py", "episodes.py", "metrics.py", "runtime.py",
               "sandbox.py", "config.py")
VERIFIER_FILES = AGENT_FILES + ("golden.py", "baseline.py", "scorer.py",
                                "eval_seeds.json")

FORBIDDEN_IN_AGENT = (
    "GoldenPicker", "golden.py", "eval_seeds", "score_submission",
    "dev.calibrate", "calibrate.py",
    "BaselinePolicy", "baseline.py",
    "_POLICY_SEED_KEY",
    # reward weights and gate constants: verifier-only (see _public_config)
    "GATE_CAP", "PERFECT_BONUS", "W_SPEED_PERFECT", "TP_BONUS_CAP_PPM",
    "W_CLEAR", "CLEAR_KNEE", "CLEAR_BASE", "CLEAR_P", "W_SPEED",
    "TP_CAP_PPM", "W_FLOOR", "W_DAMAGE", "W_BIN_HIT",
    "GOLDEN_REF_SCORE", "BASELINE_REF_SCORE",
    # the hidden seeds themselves, should they ever leak into a docstring
    "483898813954909708", "3672971350956117234", "8839858331189243848",
    "384634609664249065", "1133600575668419045", "4114120610979382378",
    "5314680192156177233", "565822075445296840", "7580374755512399761",
)

TASK_TOML = '''schema_version = "1.4"

[task]
name = "rlebench/task07-bin-clearing"
version = "1.0.0"
description = "Clear a bin of stamped steel brackets onto a conveyor drop zone with a magnet-tipped Franka Panda in MuJoCo, as fast as possible. Open-ended: ship a closed-loop policy (RGB-D in, joint targets + magnet command out); any method is legal. Scored on hidden pile seeds by clearance and speed, minus floor drops and part damage."
authors = [{ name = "Haitong Ma", email = "haitongma@g.harvard.edu" }]
keywords = ["robotics", "mujoco", "simulation", "manipulation", "bin-picking", "throughput", "custom-verifier"]

[metadata]
category = "robotics-manipulation"
difficulty = "hard"
tags = ["closed-loop-control", "rgb-d", "hidden-seeds", "cpu"]
difficulty_explanation = "Open-ended manipulation throughput: perception, motion generation and picking strategy graded end-to-end against hidden pile seeds; rewards anchored to scripted reference pickers."

[environment]
# Prebuilt by `make task07`; environment/ and tests/ remain the build inputs.
docker_image = "rlebench-task07-agent:dev"
build_timeout_sec = 1800.0
cpus = 4
memory_mb = 6144
storage_mb = 10240
network_mode = "no-network"

[agent]
timeout_sec = 14400.0
# The agent runs with the model API only; the hosts its CLI installs from stay
# with the environment.
network_mode = "allowlist"
allowed_hosts = []

[verifier]
timeout_sec = 10800.0
network_mode = "no-network"
environment_mode = "separate"

[verifier.environment]
docker_image = "rlebench-task07-verifier:dev"
cpus = 4
memory_mb = 6144
storage_mb = 10240
network_mode = "no-network"
'''

AGENT_DOCKERFILE = '''# Agent container: pinned Python + MuJoCo, headless software rendering,
# no ground truth. Built by tasks/task07/build_assets.py; pins must match
# tests/Dockerfile.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
        libgl1 libosmesa6 libegl1 ffmpeg \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \\
        mujoco==3.5.0 \\
        numpy==2.5.1 \\
        scipy==1.18.0 \\
        matplotlib==3.11.0 \\
        pytest==9.1.1 \\
        opencv-python-headless==4.13.0.92 \\
        torch==2.7.1+cpu

# deterministic software rendering — same backend the evaluation uses
ENV MUJOCO_GL=osmesa
ENV RLEBENCH_ASSETS=/workspace/assets

WORKDIR /workspace
COPY assets/ /workspace/
'''

VERIFIER_DOCKERFILE = '''# Verifier container (separate environment): full harness including the
# scorer, thresholds and evaluation seeds. test.sh locks /tests before any
# submitted code runs. Built by tasks/task07/build_assets.py.
FROM python:3.12-slim

# Submissions are re-imported here, so the runtime must match the agent
# image.
RUN apt-get update && apt-get install -y --no-install-recommends \\
        libgl1 libosmesa6 libegl1 ffmpeg \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \\
        mujoco==3.5.0 \\
        numpy==2.5.1 \\
        scipy==1.18.0 \\
        matplotlib==3.11.0 \\
        pytest==9.1.1 \\
        opencv-python-headless==4.13.0.92 \\
        torch==2.7.1+cpu

ENV MUJOCO_GL=osmesa
ENV RLEBENCH_ASSETS=/tests/assets

COPY harness/ /tests/harness/
COPY rlebench/ /tests/rlebench/
COPY assets/ /tests/assets/
COPY score_task.py test.sh /tests/

RUN chmod +x /tests/test.sh
'''

TEST_SH = '''#!/bin/bash
# Harbor verifier entry point (task07). The ONLY writer of
# /logs/verifier/reward.json.
set -u

mkdir -p /logs/verifier

# Lock the verifier context BEFORE any submitted code runs: sandboxed
# policies execute as uid 65534 and must not be able to read the scoring
# code or the evaluation seeds (deterministic piles would let readable
# seeds reconstruct ground truth), nor write the reward Harbor leaves
# world-writable.
chmod -R o-rwx,g-rwx /tests || true
chown 0:0 /logs/verifier && chmod 0755 /logs/verifier || exit 1

export PYTHONPATH=/tests
python3 /tests/score_task.py
status=$?

if [ ! -f /logs/verifier/reward.json ]; then
    echo '{"reward": 0.0, "harness_crash": 1}' > /logs/verifier/reward.json
fi

exit $status
'''

SOLVE_SH = '''#!/bin/bash
# Reference solution (Harbor Oracle) for task07: runs the baseline policy
# for a short episode through the public dev runner, then stages the
# policy package as the deliverable.
set -eu

mkdir -p /logs/artifacts
cp -r /solution/payload/policy /logs/artifacts/

cd /workspace
python3 dev_runner.py /logs/artifacts/policy --seed 35 --budget 20 || true
echo "task07 reference solution staged."
'''

def _instruction() -> str:
    return (Path(__file__).parent / "instruction.md").read_text()


def _stage_core(dst: Path) -> None:
    """rlebench.core.media, which runtime.py records episode video with."""
    core = dst / "rlebench" / "core"
    core.mkdir(parents=True)
    shutil.copy(REPO / "rlebench" / "__init__.py", dst / "rlebench" / "__init__.py")
    for name in CORE_FILES:
        shutil.copy(REPO / "rlebench" / "core" / name, core / name)


def _scan(root: Path) -> None:
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix in (".stl", ".obj", ".png", ".msh",
                                           ".mjb"):
            continue
        text = p.read_text(errors="ignore")
        for token in FORBIDDEN_IN_AGENT:
            if token in text:
                raise SystemExit(
                    f"INVARIANT VIOLATION: {token!r} found in agent-shipped "
                    f"file {p}")


AGENT_CONFIG_KEEP = ("DAMAGE_FORCE_N", "DMG_TICKS", "BIN_HIT_FORCE_N",
                     "BIN_HIT_TICKS", "BIN_HIT_REARM_TICKS")

AGENT_CONFIG = '''"""Published force limits for the bin-clearing cell.

The evaluation applies exactly these values; they are echoed in the
`cell_spec` dict your policy receives. Reward weights are not published.
"""

# Sustained (>= DMG_TICKS consecutive control ticks) contact force on a part
# above this counts as damage.
DAMAGE_FORCE_N = {DAMAGE_FORCE_N!r}
DMG_TICKS = {DMG_TICKS!r}

# Tool-bin contact above this, sustained BIN_HIT_TICKS consecutive control
# ticks, is a strike; the force must drop for BIN_HIT_REARM_TICKS before
# another can be counted.
BIN_HIT_FORCE_N = {BIN_HIT_FORCE_N!r}
BIN_HIT_TICKS = {BIN_HIT_TICKS!r}
BIN_HIT_REARM_TICKS = {BIN_HIT_REARM_TICKS!r}
'''


def _public_config(dst: Path) -> None:
    """Agent copy of config.py: only the published force limits the
    agent-side runtime needs. Reward weights, gate constants and the
    tuning summary are verifier-only."""
    ns: dict = {}
    exec(compile((SRC / "config.py").read_text(), "config.py", "exec"), ns)
    missing = [k for k in AGENT_CONFIG_KEEP if k not in ns]
    if missing:
        raise SystemExit(f"config.py is missing {missing}")
    dst.write_text(AGENT_CONFIG.format(**{k: ns[k] for k in AGENT_CONFIG_KEEP}))


def main() -> None:
    for sub in ("environment", "tests", "solution"):
        d = TASK / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    (TASK / "task.toml").write_text(TASK_TOML)

    # ----- agent environment -------------------------------------------------
    env = TASK / "environment"
    (env / "Dockerfile").write_text(AGENT_DOCKERFILE)
    assets = env / "assets"
    pkg = assets / "harness"
    pkg.mkdir(parents=True)

    for name in AGENT_FILES:
        if name == "config.py":
            _public_config(pkg / name)
        else:
            shutil.copy(SRC / name, pkg / name)
    shutil.copytree(REPO / "assets" / "robots" / "franka_emika_panda",
                    assets / "assets" / "franka_emika_panda")
    shutil.copy(SRC / "dev_runner.py", assets / "dev_runner.py")
    _stage_core(assets)
    (assets / "instruction.md").write_text(_instruction())
    _scan(assets)

    # ----- solution -----------------------------------------------------------
    sol = TASK / "solution"
    payload = sol / "payload" / "policy"
    payload.mkdir(parents=True)
    shutil.copy(SRC / "baseline.py", payload / "policy.py")
    (sol / "solve.sh").write_text(SOLVE_SH)
    (sol / "solve.sh").chmod(0o755)

    # ----- verifier -----------------------------------------------------------
    tests = TASK / "tests"
    (tests / "Dockerfile").write_text(VERIFIER_DOCKERFILE)
    (tests / "test.sh").write_text(TEST_SH)
    (tests / "test.sh").chmod(0o755)
    shutil.copy(SRC / "score_task.py", tests / "score_task.py")
    vpkg = tests / "harness"
    vpkg.mkdir(parents=True)

    for name in VERIFIER_FILES:
        shutil.copy(SRC / name, vpkg / name)
    _stage_core(tests)
    shutil.copytree(REPO / "assets" / "robots" / "franka_emika_panda",
                    tests / "assets" / "franka_emika_panda")

    print(f"built {TASK}")


if __name__ == "__main__":
    sys.exit(main())
