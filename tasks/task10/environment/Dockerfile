FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libosmesa6 libegl1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mujoco==3.5.0 numpy==2.5.1 scipy==1.18.0
ENV MUJOCO_GL=osmesa
ENV PYTHONPATH=/workspace/assets
ENV RLEBENCH_TASK10_ROBOT=/workspace/assets/robots/franka_emika_panda/panda_nohand.xml
WORKDIR /workspace
COPY assets/ /workspace/assets/
COPY dev_runner.py /workspace/dev_runner.py
RUN mkdir -p /logs/artifacts
