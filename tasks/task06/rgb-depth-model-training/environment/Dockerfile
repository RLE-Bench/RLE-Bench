# Task06 agent image: the agent gets a client, not the meshes.
# The root daemon owns MuJoCo, the robot assets, and all three block assets.
FROM python:3.12-slim

ARG AGENT_UID=1000
ARG AGENT_USER=agent
RUN test "${AGENT_UID}" -ne 0 \
    && useradd --create-home --uid ${AGENT_UID} --shell /bin/bash ${AGENT_USER}

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libosmesa6 libegl1 curl ca-certificates procps

RUN pip install --no-cache-dir mujoco==3.5.0 \
        numpy==2.5.1 \
        scipy==1.18.0 \
        matplotlib==3.11.0 \
        pytest==9.1.1 \
        torch==2.7.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

ENV MUJOCO_GL=osmesa
ENV RLEBENCH_ASSETS=/opt/private/harness/assets
ENV RLEBENCH_TRAINING_SOCKET=/run/rlebench/task06-training.sock
ENV RLEBENCH_MODALITIES=rgb,depth
ENV RLEBENCH_VARIANT=c
ENV PYTHONPATH=/workspace

COPY agent/ /workspace/
COPY private/ /opt/private/
COPY entrypoint.sh /opt/task06-entrypoint.sh

RUN chown -R root:root /opt/private \
    && chmod 700 /opt/private \
    && chmod 755 /opt/task06-entrypoint.sh \
    && mkdir -p /run/rlebench \
    && chmod 755 /run/rlebench \
    && chown -R ${AGENT_USER}:${AGENT_USER} /workspace

WORKDIR /workspace
ENTRYPOINT ["/opt/task06-entrypoint.sh"]
CMD ["sleep", "infinity"]
