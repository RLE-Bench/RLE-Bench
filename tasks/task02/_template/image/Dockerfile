ARG SIM_IMAGE=@@SIM_IMAGE@@
ARG MODELS=@@MODELS@@
FROM ${SIM_IMAGE} AS agent-base
ARG AGENT_UID=1000
RUN test "${AGENT_UID}" -ne 0 && useradd -m -u "${AGENT_UID}" -s /bin/bash agent \
    && apt-get update && apt-get install -y --no-install-recommends tini curl ca-certificates procps

FROM agent-base AS models-off
FROM agent-base AS models-on
ARG TORCH_VERSION
ARG SAM3_SHA
ARG CGN_SHA
COPY requirements.lock /opt/perception/requirements.lock
RUN test -n "$TORCH_VERSION" -a -n "$SAM3_SHA" -a -n "$CGN_SHA" \
    && pip install --no-cache-dir --no-deps --no-build-isolation -r /opt/perception/requirements.lock
COPY --from=perception sam3 /opt/perception/sam3
COPY --from=perception contact_graspnet_pytorch /opt/perception/contact_graspnet_pytorch
COPY --from=perception hf-cache /opt/perception/hf-cache
RUN pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/perception/sam3 \
    && pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/perception/contact_graspnet_pytorch
ENV PYGLET_HEADLESS=1 HF_HOME=/opt/perception/hf-cache HF_HUB_OFFLINE=1
ENV RLEBENCH_SAM3_SHA=${SAM3_SHA} RLEBENCH_CGN_SHA=${CGN_SHA}

FROM ${MODELS} AS final
COPY payload_agent /opt/rlebench
COPY payload_private /opt/private
COPY entrypoint.sh control.sh verify.sh healthcheck.sh check_isolation.py /opt/
RUN chmod 755 /opt/*.sh \
    && mkdir -p /var/lib/rlebench /run/rlebench /workspace/agent_harness /logs/artifacts /opt/private/media \
    && chown -R root:root /opt/private /opt/src /var/lib/rlebench \
    && chmod 700 /opt/private /opt/src /var/lib/rlebench \
    && chmod 755 /run/rlebench \
    && chown -R agent:agent /workspace /logs/artifacts \
    && chmod -R a+rX /opt/rlebench \
    && if [ -d /opt/perception ]; then chmod 700 /opt/perception; fi \
    && if [ -f /opt/private/HARNESS_MANUAL.md ]; then cp /opt/private/HARNESS_MANUAL.md /opt/HARNESS_MANUAL.md; fi \
    && if [ -f /opt/rlebench/harness/skills/MANUAL.md ]; then cp /opt/rlebench/harness/skills/MANUAL.md /opt/HARNESS_MANUAL.md; fi
ENV PYTHONPATH=/opt/rlebench:/workspace/agent_harness
ENV RLEBENCH_IMAGE_LEVEL=@@LEVEL@@ RLEBENCH_FAMILY=@@FAMILY@@
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/opt/entrypoint.sh"]
CMD ["sleep", "infinity"]
