# Agent container: pinned Python + MuJoCo, headless software rendering,
# no ground truth. Built by tasks/task07/build_assets.py; pins must match
# tests/Dockerfile.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libosmesa6 libegl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        mujoco==3.5.0 \
        numpy==2.5.1 \
        scipy==1.18.0 \
        matplotlib==3.11.0 \
        pytest==9.1.1 \
        opencv-python-headless==4.13.0.92 \
        torch==2.7.1+cpu

# deterministic software rendering — same backend the evaluation uses
ENV MUJOCO_GL=osmesa
ENV RLEBENCH_ASSETS=/workspace/assets

WORKDIR /workspace
COPY assets/ /workspace/
