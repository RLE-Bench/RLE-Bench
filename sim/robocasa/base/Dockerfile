# The shared RoboCasa simulator image: robosuite + robocasa at pinned commits,
# WITHOUT the asset tree. Every RLE-Bench task that needs RoboCasa derives from
# this one image (tasks/task01, tasks/task02, ...).
#
# Build it with `sim/robocasa/robocasa.sh image` (make sim-robocasa), never with a
# bare `docker build`: the pinned commits and the asset version live in ../pins.env,
# and the script passes them in as build args. The guard below fails loudly if they
# are missing.
#
# Assets are a SEPARATE versioned read-only dataset (a zstd SquashFS) mounted over
# models/assets at run time -- see each task's environment/docker-compose.yaml and
# `robocasa.sh assets`. The tree is 123k files / ~15 GB, which made every rebuild
# cost a 30+ minute 37 GB layer export while duplicating nothing useful, since
# RoboCasa only ever READS these files.
#
# Pins (CLAUDE.md invariant #3): RoboCasa needs its own mujoco/numpy/scipy, which
# conflict with the repo-wide requirements.txt. Its runtime pins are declared here.
FROM python:3.11-slim

# build-essential + linux-libc-dev are required, not optional: robosuite depends
# on pynput -> evdev, which compiles against linux/input.h, and on
# qpsolvers[quadprog], which also builds from source. Without them the build
# fails at "Failed building wheel for evdev".
#
# libegl1 is the rendering path we actually use (GPU EGL). libosmesa6 is kept as a
# CPU fallback -- see the NUMBA note near the bottom before using it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
        build-essential linux-libc-dev \
        libgl1 libegl1 libosmesa6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Single source of truth: sim/robocasa/pins.env. robosuite master and
# robocasa main both move, so a branch name is not a pin -- and no default is
# declared here, so the image's pins cannot silently fork from the venv the dev
# environment builds out of the same file.
ARG ROBOSUITE_SHA
ARG ROBOCASA_SHA
ARG ROBOCASA_ASSET_VERSION
RUN test -n "$ROBOSUITE_SHA" -a -n "$ROBOCASA_SHA" -a -n "$ROBOCASA_ASSET_VERSION" \
    || (echo "build args missing -- build with 'sim/robocasa/robocasa.sh image', which reads sim/robocasa/pins.env" >&2; exit 1)

# Sources live under /opt/src, and every later step runs from / -- NOT from the
# sources' parent directory. If cwd is the parent (e.g. WORKDIR /opt with sources
# at /opt/robosuite), `import robosuite` resolves to the source TREE as a
# namespace package instead of the installed package, and robocasa's version
# check dies with "module 'robosuite' has no attribute '__version__'".
RUN mkdir -p /opt/src \
    && git clone https://github.com/ARISE-Initiative/robosuite.git /opt/src/robosuite \
    && git -C /opt/src/robosuite checkout --quiet "$ROBOSUITE_SHA" \
    && git clone https://github.com/robocasa/robocasa.git /opt/src/robocasa \
    && git -C /opt/src/robocasa checkout --quiet "$ROBOCASA_SHA" \
    && rm -rf /opt/src/robosuite/.git /opt/src/robocasa/.git
WORKDIR /

# Pin the scientific core FIRST. robosuite's own ranges are loose
# ("numpy>=1.13.3", "mujoco>=3.3.0,<3.10"), so installing it first would let the
# resolver drift off the pins that make these tasks reproducible.
RUN pip install --no-cache-dir \
        "numpy==2.2.5" \
        "scipy==1.15.3" \
        "mujoco==3.3.1" \
        "numba==0.61.2" \
        "gymnasium==0.29.1"

# robosuite then installs against the already-satisfied pins.
RUN pip install --no-cache-dir -e /opt/src/robosuite

# robocasa's install_requires drags in the whole policy-learning stack
# (torch, torchvision, torchcodec, lerobot, tianshou, wandb, tensorboard ~ 4 GB).
# Those are imported ONLY by optional dataset/policy helpers
# (utils/lerobot_utils.py, utils/groot_utils/, utils/robomimic/,
# scripts/bench_speed.py) -- never by the env-construction path these tasks use.
# So install --no-deps and name the runtime deps explicitly.
RUN pip install --no-cache-dir --no-deps -e /opt/src/robocasa \
    && pip install --no-cache-dir \
        "Pillow==12.3.0" \
        "imageio==2.37.4" \
        "imageio-ffmpeg==0.6.0" \
        "h5py==3.16.0" \
        "lxml==6.1.1" \
        "pyyaml==6.0.3" \
        "pyyaml-include==1.4.1" \
        "termcolor==3.3.0" \
        "tqdm==4.70.0" \
        "pygame==2.6.1"

RUN python -m robocasa.scripts.setup_macros \
    && python -m robosuite.scripts.setup_macros || true

# The asset dataset is mounted over this path at run time. It MUST be the complete
# models/assets tree: 210 files here ship with the repo (120 under scenes/, 84
# under fixtures/, plus arenas, box_links, novel_instructions), and a bind mount
# OBSCURES whatever the image had at the mount point. So the SquashFS is built
# from a fully populated tree (repo files + downloads), never from the downloads
# alone -- and mounting individual heavy subdirectories is NOT safe, because
# fixtures/ mixes repo-tracked registry YAMLs with downloaded content.
ENV ROBOCASA_ASSETS=/opt/src/robocasa/robocasa/models/assets
RUN mkdir -p "$ROBOCASA_ASSETS"

# Asset dataset version this image expects. verify_assets.py compares it to the
# marker inside the mounted dataset so a stale mount fails loudly instead of
# silently changing the scenes.
ENV ROBOCASA_ASSET_VERSION=${ROBOCASA_ASSET_VERSION}

# GPU-accelerated offscreen rendering. Measured in this image (robosuite Lift,
# 3 cameras @ 128, per step):
#     OSMesa (CPU software, JIT off)   249.5 ms   ->  4.0 steps/s
#     EGL on an H100 (JIT on)           12.7 ms   -> 78.6 steps/s
# ~20x. Under EGL, 1->3 cameras costs almost nothing (10.5 -> 12.7 ms) because
# rasterization moves off the CPU and per-step cost collapses back onto physics.
# Requires GPU access at run time (the task's compose overlay pins the card).
ENV MUJOCO_GL=egl

# EGL needs TWO things the container does not get by default, and without either one
# MuJoCo silently renders on the CPU instead of failing -- measured on an L40S:
# 315 ms/step software vs 28 ms/step on the GPU, an 11x tax paid in silence. The
# giveaway is the camera pixel std, 72.75 (llvmpipe) against 72.85 (NVIDIA EGL), so
# a run can also be scored against the wrong pixels without anything looking wrong.
#
# 1. The `graphics` driver capability. The default is `compute,utility`, which gets
#    you /dev/nvidia* and nvidia-smi -- enough to look attached while libEGL_nvidia
#    is absent. Declared HERE rather than in each task's compose so the image states
#    its own requirement: the toolkit honours this even when a caller asks only for
#    `capabilities: [gpu]`, so no compose file has to know about it.
ENV NVIDIA_DRIVER_CAPABILITIES=all

# 2. The EGL vendor ICD. nvidia-container-toolkit injects the .so files but NEVER a
#    vendor .json (`nvidia-container-cli list` shows no json at all), and
#    python:3.11-slim ships only Mesa's 50_mesa.json -- so libglvnd cannot find the
#    NVIDIA driver it was just handed and falls back to llvmpipe. NVIDIA's own CUDA
#    base images carry this file, which is why the problem is invisible until you
#    build FROM something slimmer.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
       > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV PYTHONDONTWRITEBYTECODE=1

# NOTE on the OSMesa fallback: setting MUJOCO_GL=osmesa REQUIRES
# NUMBA_DISABLE_JIT=1. Debian's libOSMesa.so.8 links libLLVM.so.19.1 (llvmpipe)
# while llvmlite ships its own LLVM, and two LLVM copies in one process segfault
# the moment numba JIT-compiles -- reproducibly, in _setup_observables
# (robot.py:eef_quat_site), with rendering never touched. GALLIUM_DRIVER=softpipe,
# NUMBA_CPU_NAME=generic and NUMBA_THREADING_LAYER all fail to help. smoke.py
# asserts this pairing so it fails loudly, not as a mystery SIGSEGV. Under EGL
# there is no libOSMesa in the process, so the JIT stays on and is a straight win.

# rlebench_ro_assets is REQUIRED whenever models/assets is mounted read-only:
# RoboCasa writes a transient MJCF into the asset tree on every object
# instantiation. See the module docstring for the full story.
COPY smoke.py verify_assets.py rlebench_ro_assets.py /opt/
ENV PYTHONPATH=/opt

# Build-time check is DEPENDENCY-ONLY: the asset dataset is not present at build
# time (that is the point of this design), and `docker build` has no GPU either.
# The full env + render check runs at run time, with the dataset mounted -- see
# sim/robocasa/README.md.
ARG RUN_SMOKE=1
RUN if [ "$RUN_SMOKE" = "1" ]; then python /opt/smoke.py --deps-only; fi
