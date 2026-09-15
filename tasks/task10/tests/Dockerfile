FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libosmesa6 libegl1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mujoco==3.5.0 numpy==2.5.1 scipy==1.18.0
ENV MUJOCO_GL=osmesa
ENV RLEBENCH_TASK10_ROBOT=/tests/assets/robots/franka_emika_panda/panda_nohand.xml
COPY harness/ /tests/harness/
COPY assets/ /tests/assets/
COPY score_task.py test.sh /tests/
RUN chmod +x /tests/test.sh
