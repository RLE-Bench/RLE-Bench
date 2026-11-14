# Agent container: pinned Python + MuJoCo, headless, no ground truth.
# Pins must match requirements.txt and tests/Dockerfile.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
        mujoco==3.5.0 \
        numpy==2.5.1 \
        scipy==1.18.0 \
        matplotlib==3.11.0 \
        pytest==9.1.1

WORKDIR /workspace
COPY assets/ /workspace/
