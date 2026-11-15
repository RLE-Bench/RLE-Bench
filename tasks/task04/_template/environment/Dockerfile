# Agent image: GPU training stack, the tracking harness, the robot, the reference
# clips and the whole LAFAN1 dataset. Built from the repo root as context by
# sim/motiontrack/motiontrack.sh.
#
# The COPY list is the isolation boundary: harness modules are named one by one
# because the package also holds the scorer, the thresholds and the evaluation
# seeds, none of which may enter this image. tests/test_motiontrack_task.py
# checks the list.
#
# Versions come from sim/motiontrack/pins.env as --build-arg; the ARGs carry no
# defaults so a bare `docker build` fails rather than forking the pins.
FROM python:3.12-slim

ARG TORCH_VER
ARG TORCH_INDEX
ARG MUJOCO_VER
ARG MJWARP_VER
ARG WARP_VER
ARG ONNX_VER
ARG ONNXRUNTIME_VER
ARG ONNXSCRIPT_VER
ARG NUMPY_VER

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libosmesa6 libegl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --extra-index-url "${TORCH_INDEX}" \
        "torch==${TORCH_VER}" \
        "mujoco==${MUJOCO_VER}" \
        "mujoco-warp==${MJWARP_VER}" \
        "warp-lang==${WARP_VER}" \
        "onnx==${ONNX_VER}" \
        "onnxruntime==${ONNXRUNTIME_VER}" \
        "onnxscript==${ONNXSCRIPT_VER}" \
        "numpy==${NUMPY_VER}" \
        pytest

WORKDIR /workspace

COPY tasks/task04/harness/__init__.py \
     tasks/task04/harness/spec.py \
     tasks/task04/harness/rotations.py \
     tasks/task04/harness/robot.py \
     tasks/task04/harness/motion.py \
     tasks/task04/harness/metrics.py \
     tasks/task04/harness/mdp.py \
     tasks/task04/harness/env.py \
     tasks/task04/harness/evaluator.py \
     tasks/task04/harness/export.py \
     tasks/task04/harness/progress.py \
     /workspace/harness/

# The shared render helper the dev evaluator records with; media.py only.
COPY rlebench/__init__.py /workspace/rlebench/__init__.py
COPY rlebench/core/__init__.py rlebench/core/media.py /workspace/rlebench/core/

COPY tasks/task04/_template/environment/dev_eval.py /workspace/
COPY tasks/task04/_template/solution/payload/train.py /workspace/reference_ppo.py
COPY third_party/motiontrack/unitree_g1 /workspace/assets/unitree_g1
COPY third_party/motiontrack/lafan1 /workspace/assets/lafan1
COPY third_party/motiontrack/motions /workspace/assets/motions

ENV MUJOCO_GL=egl
ENV PYTHONPATH=/workspace
ENV MOTIONTRACK_ROBOT_DIR=/workspace/assets/unitree_g1
ENV MOTIONTRACK_DATASET=/workspace/assets/lafan1

