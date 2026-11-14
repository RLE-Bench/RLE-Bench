#!/usr/bin/env python3
"""Generate the four Harbor subtasks under ``tasks/task06/``.

All subtasks use the same blind three-shape rendering service:
  rgb-only                  RGB only  code  CPU
  rgb-depth                 RGB-D     code  CPU
  rgb-depth-model-training  RGB-D     model GPU training, CPU verification
  method-agnostic           RGB-D     code  GPU available

The agent/private/verifier/Oracle trees under each task are generated copies
and gitignored; their source lives under ``tasks/task06/harness/``. The
forbidden-token scan fails if any agent-readable file mentions verifier-side
machinery.
"""
from __future__ import annotations

import shutil
import stat
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARNESS_SRC = REPO / "tasks" / "task06" / "harness"
sys.path.insert(0, str(REPO))

FORBIDDEN_IN_AGENT = (
    "scoring", "import config", "from . import config",
    "from .. import config", "oracle", "golden", "HIDDEN_SEEDS",
    "checkpoint", "eval_seeds", "sandbox", "battery",
)

# Agent-readable modules; rendering and meshes remain in the root-only tree.
AGENT_PERCEPT_COMMON = (
    "__init__.py", "training_client.py", "evaluation_client.py",
    "public_spec.py",
)
AGENT_PERCEPT_CODE = ("estimator_template.py",)
AGENT_PERCEPT_C = ("torch_contract.py", "train_harness.py")

PINS_COMMON = ("mujoco==3.5.0", "numpy==2.5.1", "scipy==1.18.0",
               "matplotlib==3.11.0")
PIN_OPENCV = "opencv-python-headless==4.13.0.92"
PIN_SKLEARN = "scikit-learn==1.9.0"
PIN_TORCH_CPU = ("torch==2.7.1+cpu --extra-index-url "
                 "https://download.pytorch.org/whl/cpu")
PIN_TORCH_CUDA = ("torch==2.7.1+cu128 --extra-index-url "
                  "https://download.pytorch.org/whl/cu128")

VARIANTS = {
    "a": dict(
        name="rgb-only",
        title="RGB-only pose estimation",
        method="any CPU method using RGB only",
        agent_pins=PINS_COMMON + ("pytest==9.1.1", PIN_OPENCV),
        verifier_pins=PINS_COMMON + (PIN_OPENCV,),
        agent_timeout=7200, build_timeout=1800, gpu=False,
        modalities="rgb",
    ),
    "b": dict(
        name="rgb-depth",
        title="RGB-D pose estimation",
        method="any CPU method using RGB and depth",
        agent_pins=PINS_COMMON + ("pytest==9.1.1", PIN_OPENCV),
        verifier_pins=PINS_COMMON + (PIN_OPENCV,),
        agent_timeout=7200, build_timeout=1800, gpu=False,
        modalities="rgb,depth",
    ),
    "c": dict(
        name="rgb-depth-model-training",
        agent_cpus=16,
        title="learned pose estimation (20M-parameter limit)",
        method="a learned model, trained from scratch in this container",
        agent_pins=PINS_COMMON + ("pytest==9.1.1", PIN_TORCH_CUDA),
        verifier_pins=PINS_COMMON + (PIN_TORCH_CPU,),
        agent_timeout=7200, build_timeout=3600, gpu=True,
        modalities="rgb,depth",
    ),
    "d": dict(
        name="method-agnostic",
        title="pose estimation, any method",
        method="anything that works (method-agnostic)",
        agent_pins=PINS_COMMON + ("pytest==9.1.1", PIN_TORCH_CUDA,
                                  PIN_OPENCV, PIN_SKLEARN),
        verifier_pins=PINS_COMMON + (PIN_TORCH_CPU, PIN_OPENCV, PIN_SKLEARN),
        agent_timeout=7200, build_timeout=3600, gpu=True,
        modalities="rgb,depth",
    ),
}

TASK_TOML = """schema_version = "1.4"

[task]
name = "rlebench/task06-{subtask}"
version = "1.0.0"
description = "{description}"
authors = [{{ name = "Haitong Ma", email = "haitongma@g.harvard.edu" }}]
keywords = ["robotics", "mujoco", "simulation", "perception", "pose-estimation", "computer-vision"{gpu_keyword}]

[metadata]
category = "robotics-perception"
difficulty = "{difficulty}"
tags = [{tags}]
difficulty_explanation = "Blind multi-shape pose estimation under adversarial sensing, evaluated on unpublished scene seeds."

[environment]
# Prebuilt by `make task06`; environment/ and tests/ remain the build inputs.
docker_image = "rlebench-task06-{subtask}-agent:dev"
build_timeout_sec = {build_timeout}.0
cpus = {agent_cpus}
memory_mb = 6144
network_mode = "no-network"
{gpu_resource}

[agent]
timeout_sec = {agent_timeout}.0
user = "agent"
# The agent runs with the model API only; the hosts its CLI installs from stay
# with the environment.
network_mode = "allowlist"
allowed_hosts = []

[verifier]
timeout_sec = 1800.0
network_mode = "no-network"
environment_mode = "separate"

[verifier.environment]
docker_image = "rlebench-task06-{subtask}-verifier:dev"
cpus = 4
memory_mb = 6144
network_mode = "no-network"

[environment.healthcheck]
command = "test -f /run/rlebench/task06-ready"
interval_sec = 2.0
timeout_sec = 5.0
start_period_sec = 5.0
start_interval_sec = 1.0
retries = 60
"""

GPU_COMPOSE = """# The GPU of a GPU-enabled task06 agent container. Harbor reserves GPUs by count
# only and defers to this reservation, which pins the card (`rlebench run
# --device`). The GPU serves training and agent-side compute; the root-owned
# rendering service remains pinned to OSMesa.
services:
  main:
    environment:
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['${RLEBENCH_GPU:-0}']
              capabilities: [gpu]
"""

AGENT_DOCKERFILE = """# Task06 agent image: the agent gets a client, not the meshes.
# The root daemon owns MuJoCo, the robot assets, and all three block assets.
FROM python:3.12-slim

ARG AGENT_UID=1000
ARG AGENT_USER=agent
RUN test "${{AGENT_UID}}" -ne 0 \\
    && useradd --create-home --uid ${{AGENT_UID}} --shell /bin/bash ${{AGENT_USER}}

RUN apt-get update && apt-get install -y --no-install-recommends \\
        libgl1 libosmesa6 libegl1 curl ca-certificates procps

RUN pip install --no-cache-dir {pins}

ENV MUJOCO_GL=osmesa
ENV RLEBENCH_ASSETS=/opt/private/harness/assets
ENV RLEBENCH_TRAINING_SOCKET=/run/rlebench/task06-training.sock
ENV RLEBENCH_MODALITIES={modalities}
ENV RLEBENCH_VARIANT={variant}
ENV PYTHONPATH=/workspace

COPY agent/ /workspace/
COPY private/ /opt/private/
COPY entrypoint.sh /opt/task06-entrypoint.sh

RUN chown -R root:root /opt/private \\
    && chmod 700 /opt/private \\
    && chmod 755 /opt/task06-entrypoint.sh \\
    && mkdir -p /run/rlebench \\
    && chmod 755 /run/rlebench \\
    && chown -R ${{AGENT_USER}}:${{AGENT_USER}} /workspace

WORKDIR /workspace
ENTRYPOINT ["/opt/task06-entrypoint.sh"]
CMD ["sleep", "infinity"]
"""

AGENT_ENTRYPOINT = """#!/bin/sh
set -eu

socket="${RLEBENCH_TRAINING_SOCKET:-/run/rlebench/task06-training.sock}"
rm -f "${socket}" /run/rlebench/task06-ready

echo "[task06-entrypoint] starting private training renderer"
PYTHONPATH=/opt/private python -P -m harness.training_service \\
    --socket "${socket}" &
daemon_pid=$!

i=0
while [ ! -S "${socket}" ]; do
    i=$((i + 1))
    if [ "${i}" -gt 600 ]; then
        echo "[task06-entrypoint] renderer did not become ready" >&2
        exit 1
    fi
    if ! kill -0 "${daemon_pid}" 2>/dev/null; then
        echo "[task06-entrypoint] renderer exited during startup" >&2
        wait "${daemon_pid}" || true
        exit 1
    fi
    sleep 0.1
done

: > /run/rlebench/task06-ready
exec "$@"
"""

VERIFIER_DOCKERFILE = """# Verifier container (separate environment): the full harness and the
# evaluation seeds. The agent container never sees this image; at runtime
# test.sh locks /tests before any submitted code runs.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
        libgl1 libosmesa6 ffmpeg \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir {pins}

ENV MUJOCO_GL=osmesa
ENV RLEBENCH_ASSETS=/tests/harness/assets

COPY tasks/task06/{name}/tests/harness/ /tests/harness/
COPY rlebench/__init__.py /tests/rlebench/__init__.py
COPY rlebench/core/__init__.py rlebench/core/model.py rlebench/core/scoring.py rlebench/core/media.py /tests/rlebench/core/
COPY tasks/task06/{name}/tests/score_task.py tasks/task06/{name}/tests/test.sh /tests/

RUN chmod +x /tests/test.sh
"""

TEST_SH = """#!/bin/bash
# Harbor verifier entry point ({name}); the only writer of
# /logs/verifier/reward.json.
set -u

mkdir -p /logs/verifier

# Lock the verifier context BEFORE any submitted code can run: sandboxed
# estimators execute as uid 65534 and must not be able to read the scoring
# code, thresholds, or evaluation seeds (deterministic rendering would let
# readable seeds reconstruct ground truth), nor write the reward Harbor
# leaves world-writable.
chmod -R o-rwx,g-rwx /tests || true
chown 0:0 /logs/verifier && chmod 0755 /logs/verifier || exit 1

export PYTHONPATH=/tests
python3 /tests/score_task.py
status=$?

if [ ! -f /logs/verifier/reward.json ]; then
    echo '{{"reward": 0.0, "harness_crash": 1}}' > /logs/verifier/reward.json
fi

exit $status
"""

SCORE_TASK_PY = '''"""Task06 / {task_name} verifier scoring entry point (runs inside the verifier container).

Stages the agent's artifacts as UNTRUSTED input, scores variant "{v}"
through the sandboxed pipeline, and writes reward.json / report.json.
"""
from __future__ import annotations

import json
import os
import shutil
import traceback

VARIANT = "{v}"
ARTIFACTS = os.environ.get("RLEBENCH_ARTIFACTS", "/logs/artifacts")
STAGING = os.environ.get("RLEBENCH_STAGING", "/tmp/submission")
VERIFIER_DIR = os.environ.get("RLEBENCH_VERIFIER_DIR", "/logs/verifier")


def _write(reward, report=None):
    os.makedirs(VERIFIER_DIR, exist_ok=True)
    with open(os.path.join(VERIFIER_DIR, "reward.json"), "w") as f:
        json.dump(reward, f, indent=2)
    if report is not None:
        with open(os.path.join(VERIFIER_DIR, "report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)


def _media():
    """Stage B episode videos for a human reader; never an input to the score."""
    try:
        from rlebench.core.media import Media
        return Media.verifier(VERIFIER_DIR)
    except Exception as e:  # noqa: BLE001
        print(f"media disabled: {{e}}")
        return None


def main():
    try:
        if os.path.exists(STAGING):
            shutil.rmtree(STAGING)
        from harness.artifact_staging import stage_submission
        stage_submission(ARTIFACTS, STAGING)

        from harness.scorer import score_submission
        media = _media()
        report = score_submission(STAGING, VARIANT, media=media)

        reward = {{"reward": float(report["reward"]),
                   "raw_total": float(report["raw_total"]),
                   "gated": int(report["gated"])}}
        for stage, val in report["stages"].items():
            reward[f"stage_{{stage}}"] = float(val)
        for cid, val in report["checkpoints"].items():
            reward[f"cp_{{cid}}"] = float(val)
        for k in ("abs_median_trans_mm", "abs_median_rot_deg",
                  "stage_b_median_trans_mm", "stage_b_median_rot_deg",
                  "abs_occluded_trans_mm", "frame_time_s",
                  "stage_a_frames", "stage_b_frames",
                  "stage_b_occluded_frames", "stage_b_occluded_fraction",
                  "stage_a_shape_accuracy", "stage_b_shape_accuracy",
                  "accuracy_reward", "efficiency_deduction",
                  "inference_wall_s", "inference_hz"):
            if report.get(k) is not None:
                reward[k] = float(report[k])
        _write(reward, report)
        print(json.dumps(report, indent=2, default=str))
        if media is not None:
            media.report()
    except Exception:
        traceback.print_exc()
        _write({{"reward": 0.0, "scoring_error": 1}})


if __name__ == "__main__":
    main()
'''

SOLVE_SH_CODE = """#!/bin/bash
# Reference solution (Harbor Oracle) for {task_name}.
set -eu

mkdir -p /logs/artifacts
cp -r /solution/payload/. /logs/artifacts/
echo "{task_name} reference solution staged."
"""

SOLVE_SH_C = """#!/bin/bash
# Reference solution (Harbor Oracle) for rgb-depth-model-training: trains the
# reference CNN in this container on the provided GPU, then stages model.pt
# and its training report.
set -eu

mkdir -p /logs/artifacts
cd /workspace

python3 /solution/train_solution.py /logs/artifacts
echo "rgb-depth-model-training reference solution staged."
"""

TRAIN_SOLUTION_PY = '''"""Reference rgb-depth-model-training solution: train a small CNN.

Runs in the agent container (GPU). Dataset generation and epoch-controlled
optimization use the shipped harness API.
"""
import json
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, "/workspace")

from harness import spec, torch_contract, train_harness
sys.path.insert(0, "/solution/payload")
from rbp.reference_labels import label_shape


class PoseNet(nn.Module):
    """Calibrated object-centric pose network."""

    def __init__(self):
        super().__init__()
        self.crop_size = 160
        self.crop_half_pixels = 112.0
        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.net = nn.Sequential(
            block(6, 32, 2), block(32, 64, 2), block(64, 96, 2),
            block(96, 128, 2), block(128, 160, 2),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(
            nn.Linear(166, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 4))
        self.shape_head = nn.Linear(166, 3)
        self.fx = float(spec.intrinsics()[0, 0])
        self.fy = float(spec.intrinsics()[1, 1])
        self.cx = float(spec.intrinsics()[0, 2])
        self.cy = float(spec.intrinsics()[1, 2])
        self.register_buffer(
            "camera_rotation",
            torch.as_tensor(spec.camera_rotation(), dtype=torch.float32))
        self.register_buffer(
            "camera_position",
            torch.as_tensor(spec.CAM_POS, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        red, green, blue = x[:, 0], x[:, 1], x[:, 2]
        mask = ((red > (60.0 / 255.0))
                & (red > 1.35 * green)
                & (red > 1.35 * blue)).to(dtype=x.dtype)

        rows = torch.arange(height, dtype=x.dtype, device=x.device)
        cols = torch.arange(width, dtype=x.dtype, device=x.device)
        mass = mask.sum(dim=(1, 2)).clamp_min(1.0)
        u = (mask.sum(dim=1) * cols.unsqueeze(0)).sum(dim=1) / mass
        v = (mask.sum(dim=2) * rows.unsqueeze(0)).sum(dim=1) / mass

        du = (cols.unsqueeze(0) - u.unsqueeze(1)) / self.crop_half_pixels
        dv = (rows.unsqueeze(0) - v.unsqueeze(1)) / self.crop_half_pixels
        col_mass = mask.sum(dim=1)
        row_mass = mask.sum(dim=2)
        var_u = (col_mass * du.square()).sum(dim=1) / mass
        var_v = (row_mass * dv.square()).sum(dim=1) / mass
        cov_uv = (
            mask * dv.unsqueeze(2) * du.unsqueeze(1)
        ).sum(dim=(1, 2)) / mass

        axis = torch.linspace(-1.0, 1.0, self.crop_size,
                              dtype=x.dtype, device=x.device)
        grid_u = (u[:, None, None]
                  + self.crop_half_pixels * axis[None, None, :])
        grid_v = (v[:, None, None]
                  + self.crop_half_pixels * axis[None, :, None])
        grid_u = grid_u.expand(batch, self.crop_size, self.crop_size)
        grid_v = grid_v.expand(batch, self.crop_size, self.crop_size)
        grid = torch.stack((2.0 * grid_u / float(width - 1) - 1.0,
                            2.0 * grid_v / float(height - 1) - 1.0), dim=-1)
        crop_input = torch.cat((x, mask.unsqueeze(1)), dim=1)
        crop = F.grid_sample(crop_input, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=True)
        visual = self.net(crop).flatten(1)

        d_camera = torch.stack(((u - self.cx) / self.fx,
                                -(v - self.cy) / self.fy,
                                -torch.ones_like(u)), dim=1)
        d_world = d_camera @ self.camera_rotation.t()
        scale = ((spec.BLOCK_HEIGHT / 2.0 - self.camera_position[2])
                 / d_world[:, 2].clamp_max(-1e-6))
        center_xy = (self.camera_position[:2].unsqueeze(0)
                     + scale.unsqueeze(1) * d_world[:, :2])

        geometry = torch.stack((u / float(width) - 0.5,
                                v / float(height) - 0.5,
                                mass / float(height * width) * 50.0,
                                var_u, cov_uv, var_v), dim=1)
        features = torch.cat((visual, geometry), dim=1)
        raw = self.head(features)
        xy = center_xy + 0.15 * torch.tanh(raw[:, :2])
        return torch.cat((xy, raw[:, 2:4], self.shape_head(features)), dim=1)

def main():
    out = sys.argv[1]
    data_dir = "/tmp/dataset"
    print("generating dataset ...")
    train_harness.generate_dataset(data_dir, seeds=spec.DESIGN_SEEDS,
                                   frames_per_seed=300, episode_frames=100)
    torch.manual_seed(0)
    model = PoseNet()
    print("training for 54 epochs ...")
    report = train_harness.fit(model, data_dir, epochs=54,
                               batch_size=32, lr=1e-3, seed=0,
                               report_path=out + "/training_report.json",
                               shape_labeler=label_shape)
    print(json.dumps(report, indent=2))
    torch_contract.save_model(model, out + "/model.pt")

    print("payload staged")


if __name__ == "__main__":
    main()
'''

INSTRUCTION_COMMON = """A robot-mounted stick pushes one of three red, asymmetric blocks on a table.
From a fixed oblique camera, estimate the block's planar pose and identify its
shape. Predictions should remain accurate through rotation and occlusion,
with efficient CPU inference.

## Training data

Use the public clients in `/workspace/harness`. Meshes and the private scene
are not available; learn shape geometry and orientation conventions from
rendered examples.

~~~python
from harness import episodes

frame = episodes.single_frame(11)
print(frame.obs.keys())  # __KEYS__
print(frame.gt)          # noisy (x, y, theta), no shape label
episode = episodes.push_episode(23)
~~~

Public design seeds are 11, 23, and 37. You may draw additional 32-bit seeds
from them. Calls are deterministic by seed, including observation and label
noise. Pose labels are noisy; shape labels are not provided. The shape stays
fixed within an episode but may differ between scenes.

Observations contain __OBS__.
RGB is uint8; depth, when available, is in meters with NaN for missing pixels.
`K` is the camera intrinsic matrix, `T_cam_table` its pose in the table frame,
and `t` is time in seconds. See `harness/spec.py` for camera constants.

Pose means `(x, y, theta)`: x and y in meters in the table frame, theta about
+z in radians wrapped to (-pi, pi]. Each block is 40 mm tall. Its orientation
convention is defined by the pose labels. Shape IDs are fixed: 0 = T,
1 = C, 2 = F. Use this mapping on every frame, including after resets.
Shape identity is never an input.

## Method and submission

__METHOD__

__OUTPUT__

All variants must run on CPU during evaluation, even when a GPU is available
for training. Process observations causally, without future frames. Return
finite outputs and handle missing depth and temporary occlusion gracefully.

## Development

After staging a submission, `harness.evaluator.evaluate()` provides limited
design-set pose diagnostics. At most six calls are available per container.
Training and data generation must finish within the two-hour agent budget.
Only files under `/logs/artifacts/` are collected; reports and CSVs are optional.
"""

CODE_DELIVERABLE = """Submit `/logs/artifacts/estimator.py` with `make_estimator()`, returning an
object with `reset()` and `update(*, rgb, K, T_cam_table, t, depth=None)`.
Each update returns exactly `(x, y, theta, shape_id)`, with an integer shape ID.
Reset clears tracking state, not the shape-ID convention. See
`harness/estimator_template.py`. Helper files may sit beside `estimator.py`;
the harness package is not importable during evaluation."""

MODEL_DELIVERABLE = """Submit `/logs/artifacts/model.pt`, a TorchScript module saved on CPU using
`harness.torch_contract.save_model`. No submission Python executes during
evaluation. Each frame is inferred independently, without persistent state.

Input is `[5,H,W]` float32: RGB scaled to 0..1, depth in meters with NaN
replaced by zero, and a depth-validity mask. Output is `[7]`:
`(x, y, cos(theta), sin(theta), T_logit, C_logit, F_logit)`.
The shape ID is the argmax of the three logits; ties select the lowest index.
Use `harness/torch_contract.py` for tensor packing and serialization.
The module may contain at most 20,000,000 tensor elements across parameters,
buffers, and frozen constants. `harness/train_harness.py` provides training
utilities."""


def _readme(v: str, cfg: dict) -> str:
    hardware = "1 NVIDIA GPU" if cfg["gpu"] else "CPU only"
    deliverable = "`model.pt` (TorchScript)" if v == "c" else "`estimator.py`"
    gate = ("the TorchScript module loads within the 20M-element limit"
            if v == "c" else "the estimator loads and returns a pose")
    return f"""# Task 06 / {cfg["name"]}: {cfg["title"]}

Estimate the planar pose of an unlabeled T, C or F block from a fixed camera
and identify its shape; {cfg["method"]}. See [instruction.md](instruction.md)
for the contract and the family [README](../README.md) for the scoring.

The agent container runs Python 3.12 with MuJoCo 3.5, numpy and scipy on
{cfg.get('agent_cpus', 4)} CPUs, 6144 MB RAM, {hardware}, with a
{cfg["agent_timeout"]}-second session. The workspace holds clients for the
root-owned renderer and the design diagnostic; meshes, shape identity, exact
labels, seeds and grading code stay private. The deliverable is {deliverable};
the gate is that {gate}.

## Running it

~~~bash
make task06-assets
harbor run -p tasks/task06/{cfg["name"]} -a oracle
rlebench run task06/{cfg["name"]} -a <agent> -m <model>  # adds the agent's install hosts
~~~
"""


def _scan(root: Path) -> None:
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix in (".stl", ".obj", ".png", ".msh"):
            continue
        text = p.read_text(errors="ignore")
        for token in FORBIDDEN_IN_AGENT:
            if token in text:
                raise SystemExit(
                    f"INVARIANT VIOLATION: {token!r} found in agent-shipped "
                    f"file {p}")


# The core modules the scorers import; the same list the verifier Dockerfile
# COPYs, so the rest of rlebench.core never rides into an image.
CORE_FILES = ("__init__.py", "model.py", "scoring.py")


def _stage_core(dst_root: Path) -> None:
    """rlebench.core (scoring + model validity) that the scorers import."""
    rl = dst_root / "rlebench"
    (rl / "core").mkdir(parents=True)
    shutil.copy(REPO / "rlebench" / "__init__.py", rl / "__init__.py")
    for name in CORE_FILES:
        shutil.copy(REPO / "rlebench" / "core" / name, rl / "core" / name)


def build_variant(v: str) -> None:
    cfg = VARIANTS[v]
    task = REPO / "tasks" / "task06" / cfg["name"]
    if task.exists():
        shutil.rmtree(task)
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()

    description = (
        "Estimate the planar pose (x, y, theta) of one of three unlabeled "
        "asymmetric blocks from a fixed oblique camera; neither meshes nor "
        f"shape identity are exposed — {cfg['method']}."
    )
    (task / "task.toml").write_text(TASK_TOML.format(
        subtask=cfg["name"], description=description,
        build_timeout=cfg["build_timeout"],
        agent_timeout=cfg["agent_timeout"],
        agent_cpus=cfg.get("agent_cpus", 4),
        gpu_resource="gpus = 1" if cfg["gpu"] else "",
        gpu_keyword=', "gpu"' if cfg["gpu"] else "",
        difficulty="hard" if cfg["gpu"] else "medium",
        tags=", ".join(f'"{t}"' for t in
                       ("pose-estimation", "hidden-seeds",
                        "gpu" if cfg["gpu"] else "cpu"))))

    # ----- agent environment -------------------------------------------------
    env = task / "environment"
    pins = " \\\n        ".join(cfg["agent_pins"])
    inst = _instruction(v, cfg)
    # Every subset uses the private blind-shape renderer.
    (env / "Dockerfile").write_text(
        AGENT_DOCKERFILE.format(
            pins=pins, modalities=cfg["modalities"], variant=v))
    (env / "entrypoint.sh").write_text(AGENT_ENTRYPOINT)
    (env / "entrypoint.sh").chmod(
        (env / "entrypoint.sh").stat().st_mode | stat.S_IXUSR)
    if cfg["gpu"]:
        (env / "docker-compose.yaml").write_text(GPU_COMPOSE)

    agent_root = env / "agent"
    pkg = agent_root / "harness"
    pkg.mkdir(parents=True)
    
    (pkg / "__init__.py").write_text("")
    shutil.copy(HARNESS_SRC / "training_client.py",
                pkg / "episodes.py")
    shutil.copy(HARNESS_SRC / "evaluation_client.py",
                pkg / "evaluator.py")
    shutil.copy(HARNESS_SRC / "public_spec.py",
                pkg / "spec.py")
    if v == "c":
        for name in AGENT_PERCEPT_C:
            shutil.copy(HARNESS_SRC / name,
                        pkg / name)
    else:
        for name in AGENT_PERCEPT_CODE:
            shutil.copy(HARNESS_SRC / name,
                        pkg / name)
    (agent_root / "instruction.md").write_text(inst)

    private = env / "private"
    private.mkdir(parents=True)
    shutil.copytree(
        HARNESS_SRC, private / "harness",
        ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(
        REPO / "assets" / "robots" / "franka_emika_panda",
        private / "harness" / "assets" / "franka_emika_panda")
    _stage_core(private)
    _scan(agent_root)
    (task / "instruction.md").write_text(inst)
    (task / "README.md").write_text(_readme(v, cfg))

    # ----- verifier ------------------------------------------------------------
    tests = task / "tests"
    vpins = " \\\n        ".join(cfg["verifier_pins"])
    (tests / "Dockerfile").write_text(VERIFIER_DOCKERFILE.format(pins=vpins, name=cfg["name"]))
    (tests / "test.sh").write_text(TEST_SH.format(name=f"task06-{cfg['name']}"))
    (tests / "test.sh").chmod((tests / "test.sh").stat().st_mode
                              | stat.S_IXUSR)
    (tests / "score_task.py").write_text(SCORE_TASK_PY.format(v=v, task_name=cfg["name"]))
    shutil.copytree(HARNESS_SRC, tests / "harness",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "assets" / "robots" / "franka_emika_panda",
                    tests / "harness" / "assets" / "franka_emika_panda")

    # ----- solution ---------------------------------------------------------------
    sol = task / "solution"
    if v == "c":
        (sol / "solve.sh").write_text(SOLVE_SH_C)
        (sol / "train_solution.py").write_text(TRAIN_SOLUTION_PY)
        from harness.reference import build_reference_submission
        build_reference_submission("d", str(sol / "payload"))
    else:
        (sol / "solve.sh").write_text(SOLVE_SH_CODE.format(v=v, task_name=cfg["name"]))
        from harness.reference import build_reference_submission
        payload = sol / "payload"
        build_reference_submission(v, str(payload))
    (sol / "solve.sh").chmod((sol / "solve.sh").stat().st_mode
                             | stat.S_IXUSR)
    # Emitted permissions must not depend on the caller's umask or source modes.
    for path in task.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() or path.suffix == ".sh" else 0o644)
    print(f"{cfg['name']}: built")


def _instruction(v: str, cfg: dict) -> str:
    methods = {
        "a": "Use RGB only with numpy, scipy, and OpenCV on CPU. Stateful tracking is allowed.",
        "b": "Use RGB-D with numpy, scipy, and OpenCV on CPU. Stateful tracking is allowed.",
        "c": "Train a model from scratch in this container. One GPU is available for training.",
        "d": "Any method is allowed using numpy, scipy, OpenCV, torch, and scikit-learn. "
             "One GPU is available during development; stateful tracking is allowed.",
    }
    keys = "rgb, K, T_cam_table, t" if v == "a" else "rgb, depth, K, T_cam_table, t"
    body = (INSTRUCTION_COMMON.replace("__KEYS__", keys)
            .replace("__OBS__", f"`{keys}`")
            .replace("__METHOD__", methods[v])
            .replace("__OUTPUT__", MODEL_DELIVERABLE if v == "c" else CODE_DELIVERABLE))
    return f"# Task 06 / {cfg['name']} — Pose and shape estimation\n\n" + body


def main():
    for v in VARIANTS:
        build_variant(v)
    print("all task06 subtasks built")


if __name__ == "__main__":
    main()
