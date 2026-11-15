# Agent container: pinned Python + MuJoCo, headless, no ground truth.
# Pins must match tests/Dockerfile.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        mujoco==3.5.0 \
        numpy==2.5.1 \
        scipy==1.18.0 \
        matplotlib==3.11.0 \
        pytest==9.1.1

WORKDIR /workspace
# Public shelf data and chassis-free stock components only. The agent creates
# robot.xml from scratch; evaluation code remains in the separate verifier.
COPY assets/ /workspace/
