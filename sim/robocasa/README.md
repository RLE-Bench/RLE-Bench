# The RoboCasa simulator layer

One copy of everything that is "RoboCasa", shared by every task that needs it
(currently `tasks/task01` and `tasks/task02`). Nothing here is task-specific: no
scoring, no thresholds, no task lists, no harness package. Deliberately — a
task image derives FROM this one, and anything staged here would be inherited by
all of them.

```
sim/robocasa/
├── README.md           this file
├── pins.env            the commits, asset version and image tag — ONE copy
├── eval_video.py       the evaluation-trial video recorder every RoboCasa family's
│                       stager syncs into its harness package (never baked in here)
└── base/
    ├── Dockerfile              robosuite + robocasa at the pinned commits
    ├── smoke.py                does it import, construct, render?
    ├── verify_assets.py        is the mounted dataset present, complete, current?
    └── rlebench_ro_assets.py  makes RoboCasa work against a READ-ONLY asset tree
```

Everything you *do* with this layer — build the image, get the dataset, stand up a
dev environment — is one script: [`sim/robocasa/robocasa.sh`](robocasa.sh).
It lives outside this directory because it is host-side tooling, not part of what a
task image is built from. `sim/robocasa/robocasa.sh help` lists every command; three of
them have Makefile targets. Dev mode populates:

```
third_party/                gitignored, ~23 GB on disk
├── robosuite/              at ROBOSUITE_SHA
└── robocasa/               at ROBOCASA_SHA
    └── robocasa/models/assets/   the dataset + its ROBOCASA_ASSET_VERSION marker
.venv-robocasa/             the venv that imports them
```

In-tree on purpose: a checkout is self-describing, so every Makefile default,
container mount and test invocation derives from the repo's own location and a
fresh clone needs no configuration.

Future simulators get a sibling directory (`sim/<name>/`).

## Two modes

**User mode — you want to run a task.** You need the simulator image and the
asset dataset. `make sim-robocasa` provides both (the dataset lands at
`third_party/robocasa/robocasa/models/assets`), and `rlebench prepare task01|task02`
runs it first. The `rlebench` run lanes point the compose overlay at that path unless
`ROBOCASA_ASSET_DIR` is set; a direct `harbor run` needs the export:

```sh
make sim-robocasa                          # simulator image + dataset (+ dev venv)
make task02                                # the task's agent images

ROBOCASA_ASSET_DIR=$PWD/third_party/robocasa/robocasa/models/assets \
    harbor run -p tasks/task02 -i 01-washing-dishes -a oracle --override-gpus 0 --yes
```

A dataset elsewhere (a SquashFS from `assets mount`, a shared copy) is used by
pointing `ROBOCASA_ASSET_DIR` at it — the fixed path is the default, not a requirement.

**Dev mode — you want to change the harness.** Adds the vendored sources, the
asset dataset and a venv importing them, which the simulator-marked tests and any
host-side experiment need:

```sh
make sim-robocasa                    # image + third_party/ + .venv-robocasa (~15 GB download)
MUJOCO_GL=egl make test TASK=task02 PY=.venv-robocasa/bin/python
```

Idempotent — it reuses the venv, re-points the editable installs and skips a dataset
that is already current. Stages can be run alone through the script (`ARGS` reaches
its `setup`):

```sh
make sim-robocasa ARGS=--no-assets         # sources and venv only, no dataset
make sim-robocasa ARGS=--force-assets      # re-download the dataset
sim/robocasa/robocasa.sh assets            # the dataset by itself
```

RoboCasa's pins (mujoco 3.3.1) conflict with the repo-wide `requirements.txt`
(3.5.0), which is why this is a separate venv rather than `make install`.

## The pins

`pins.env` is the single source of truth, read by three consumers that would
otherwise drift:

| consumer | how |
| --- | --- |
| `Makefile` | `include`s it, passes `--build-arg` |
| `base/Dockerfile` | declares the ARGs with **no defaults** — a bare `docker build` fails loudly |
| `sim/robocasa/robocasa.sh` | `source`s it to check out the same commits and stamp the dataset |

Its syntax must stay valid for both make and sh: `KEY=value`, no spaces, no
quotes.

Bump deliberately. robosuite `master` and robocasa `main` both move, so a branch
name is not a pin, and task02's stage functions are hand-written mirrors of
RoboCasa source — re-run `tests/runtime/test_stage_credit.py` with `RLEBENCH_TEST_TASK=task02` after any bump. Also audit the mirrored predicates against the new source before accepting
the pin change.

## The asset dataset

123,434 files / ~15 GB that RoboCasa only ever **reads**, so it ships as a
separately versioned read-only dataset rather than image content. Baking it in
cost a 30+ minute 37 GB layer export on every rebuild and duplicated nothing
useful.

One subcommand each:

```sh
sim/robocasa/robocasa.sh assets            # download (the default) -> third_party/…/models/assets
sim/robocasa/robocasa.sh assets verify     # present, complete, current?
sim/robocasa/robocasa.sh assets pack       # a populated tree -> .sqfs + .sha256 (~9.6 GB)
sim/robocasa/robocasa.sh assets mount      # checksum, then mount read-only (needs root)
sim/robocasa/robocasa.sh assets umount
```

`pack` and `mount` need `ROBOCASA_ASSET_HOME` — where the packed `.sqfs` is kept. It
is site policy, so it has no default and the script says so rather than failing on a
broken path.

The checksum is verified **first**, on purpose: a silently wrong asset version
changes the scenes with no error anywhere.

Two rules the packing step depends on:

- The source must be the **complete merged tree** — repo-tracked files *plus*
  downloads. 210 files under `models/assets` ship with the robocasa repo (120
  under `scenes/`, 84 under `fixtures/`, plus `arenas`, `box_links`,
  `novel_instructions`), and a bind mount **obscures** whatever the image had at
  the mount point.
- Mounting individual heavy subdirectories is **not** safe: `fixtures/` mixes
  repo-tracked registry YAMLs with downloaded content.

`verify_assets.py` asserts all of this at container start, and
`sim/robocasa/robocasa.sh` runs it over the populated tree before it reports
success.

**Nothing upstream writes the `ROBOCASA_ASSET_VERSION` marker** — RoboCasa's
downloader does not know this repo's versioning. `robocasa.sh` writes it,
last, so its presence means "download complete". Populate a tree any other way
and you must stamp it yourself, or every guard that reads it fails.

No root? `unsquashfs -d <dir> <sqfs>` materializes the tree as ordinary files.
Correct, but gives up what SquashFS buys (one file instead of 123k host inodes).

## Checks

Both need a GPU and the mounted dataset.

```sh
ASSETS=-v"$ROBOCASA_ASSET_DIR:/opt/src/robocasa/robocasa/models/assets:ro"

# Does the image actually simulate? The BUILD-time smoke test is deps-only —
# `docker build` has neither the dataset nor a GPU.
docker run --rm --gpus all $ASSETS rlebench-robocasa-sim:1.0.1 \
    sh -c "python /opt/verify_assets.py && python /opt/smoke.py"
```

`smoke.py` is also the **orientation** check, which no unit test can make: it
asserts that an on-demand render and the streamed observation agree, and writes
one frame out so a human can confirm both are the right way up. They disagreed in
production once — the agent's frames were inverted relative to every saved
artifact — and nothing raised, because an upside-down image is still a valid
image.

Per-task adversarial checks (can the agent reach the simulator off-meter? can it
discover the evaluation split?) live in each task's `environment/check_*.sh`;
their headers carry the exact invocation.

## Rendering

`MUJOCO_GL=egl`, and it is worth it. Measured in this image (robosuite Lift,
3 cameras @ 128, per step):

| backend | per step | throughput |
| --- | --- | --- |
| OSMesa (CPU software, JIT off) | 249.5 ms | 4.0 steps/s |
| EGL on an H100 (JIT on) | 12.7 ms | 78.6 steps/s |

Under EGL, going from 1 to 3 cameras costs almost nothing (10.5 → 12.7 ms):
rasterization moves off the CPU and per-step cost collapses back onto physics.

**EGL needs two things the container does not get by default**, and missing either
one means MuJoCo renders in *software* without saying so. Both are baked into
`base/Dockerfile`, so nothing on the host or in any compose file has to know:

| what | why |
| --- | --- |
| `ENV NVIDIA_DRIVER_CAPABILITIES=all` | the default `compute,utility` gives you `/dev/nvidia*` and `nvidia-smi` — enough to look attached while `libEGL_nvidia.so` is absent |
| `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` | nvidia-container-toolkit injects the `.so` files but **never** a vendor `.json`, and `python:3.11-slim` ships only Mesa's — so libglvnd cannot find the driver it was handed |

Measured on an L40S, RoboCasa kitchen scene: **315 ms/step** on the silent
llvmpipe fallback against **28 ms/step** on the GPU. The tell is the camera pixel
std — 72.75 for llvmpipe, 72.85 for NVIDIA EGL — which means a vision agent can
also be *scored against different pixels* with nothing raising. `smoke.py` asserts
the ICD is present whenever `MUJOCO_GL=egl`, so the fallback fails loudly instead.

**The OSMesa fallback requires `NUMBA_DISABLE_JIT=1`.** Debian's `libOSMesa.so.8`
links `libLLVM.so.19.1` while llvmlite ships its own LLVM, and two LLVM copies in
one process segfault the moment numba JIT-compiles — reproducibly, in
`_setup_observables`, with rendering never touched. `smoke.py` asserts the
pairing so it fails loudly instead of as a mystery SIGSEGV.
