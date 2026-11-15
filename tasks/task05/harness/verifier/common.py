"""Shared trusted verifier helpers for the task05 nanoVLA subtasks.

Phases: replay `solution.py` in train mode as the training UID, freeze what it
wrote, start `solution.py` in serve mode as the serving UID, roll episodes out
through the policy socket as root, score. The serving UID cannot reach the
shards (their parent directory is group-only for the training UID) nor any
scratch the training phase left behind (swept by root in between).
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, NoReturn

PROJECT = Path("/workspace/nanovla")
VERIFIER_CODE = Path("/opt/nanovla-verifier")
ARTIFACT_DIR = Path("/logs/nanovla-handoff/submission")
ARTIFACT = ARTIFACT_DIR / "solution.py"
STAGING_DIR = Path("/tmp/nanovla-submission")
STAGED_SOLUTION = STAGING_DIR / "solution.py"
OUTPUT_DIR = Path("/tmp/nanovla-output")
FROZEN_DIR = Path("/tmp/nanovla-frozen")
PRIVATE_DIR = Path("/tmp/nanovla-verifier")
IPC_DIR = Path("/tmp/nanovla-ipc")
# Operator-supplied resume point: a directory holding the frozen ckpt.pt (and
# meta.json) of an interrupted run plus its <name>.jsonl.inc rows. Set only by
# the harness operator through the verifier environment; a submission never
# sees it, because _submission_env builds its environment from scratch.
RESUME_DIR = Path(os.environ["NANOVLA_RESUME_DIR"]) if os.environ.get("NANOVLA_RESUME_DIR") else None
VERIFIER_DIR = Path("/logs/verifier")
HF_BUNDLE = Path("/assets/hf")
SHARDS_PARENT = Path("/assets/nanovla")
ROBOTWIN = Path("/opt/robotwin")
SIMULATOR = os.environ.get("NANOVLA_SIMULATOR", "libero")
ACTION_DIM = {"robotwin": "14"}.get(SIMULATOR, "7")
SCRATCH_ROOTS = (Path("/tmp"), Path("/dev/shm"))
MAX_SOLUTION_BYTES = 1024 * 1024
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
MAX_META_BYTES = 1024 * 1024
MAX_PARAMETERS = 2_000_000_000
TRAIN_UID = 65534
SERVE_UID = 65533
OUTPUT_FILES = ("ckpt.pt", "meta.json")


class ScoreFailure(RuntimeError):
    pass


def fail(reason: str) -> NoReturn:
    raise ScoreFailure(reason)


def _reset_dir(path: Path, mode: int = 0o755) -> None:
    if path.is_symlink() or path.exists():
        _remove(path)
    path.mkdir(parents=True, mode=mode)
    path.chmod(mode)


def _remove(path: Path) -> None:
    """Delete a submission-owned tree. The verifier has CAP_CHOWN but no DAC
    override, so root first claims every entry it needs to write to."""
    if path.is_dir() and not path.is_symlink():
        if os.geteuid() == 0:
            with contextlib.suppress(OSError):
                os.chown(path, 0, 0, follow_symlinks=False)
                os.chmod(path, 0o700)
            for root, dirs, files in os.walk(path, topdown=True):
                for name in dirs:
                    entry = os.path.join(root, name)
                    with contextlib.suppress(OSError):
                        os.chown(entry, 0, 0, follow_symlinks=False)
                        os.chmod(entry, 0o700)
                for name in files:
                    with contextlib.suppress(OSError):
                        os.chown(os.path.join(root, name), 0, 0, follow_symlinks=False)
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


# ----------------------------------------------------------------- submission

def validate_submission() -> str:
    if not ARTIFACT_DIR.is_dir():
        fail("missing finalized submission handoff")
    entries = sorted(entry.name for entry in ARTIFACT_DIR.iterdir())
    if entries != ["solution.py"]:
        fail(f"submission directory must contain only solution.py; found {entries}")
    info = ARTIFACT.lstat()
    if not stat.S_ISREG(info.st_mode) or ARTIFACT.is_symlink():
        fail("solution.py must be a regular file, not a symlink")
    if not 1 <= info.st_size <= MAX_SOLUTION_BYTES:
        fail(f"solution.py size must be 1..{MAX_SOLUTION_BYTES} bytes")
    try:
        source = ARTIFACT.read_text(encoding="utf-8")
        compile(source, "solution.py", "exec")
    except (UnicodeDecodeError, SyntaxError) as exc:
        fail(f"solution.py is not valid UTF-8 Python: {exc}")
    _reset_dir(STAGING_DIR)
    STAGED_SOLUTION.write_text(source, encoding="utf-8")
    STAGED_SOLUTION.chmod(0o444)
    return source


# ------------------------------------------------------------------ processes

def _drop_privileges(uid: int) -> Callable[[], None]:
    def drop() -> None:
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)
        os.umask(0o077)
        with contextlib.suppress(Exception):
            libc = ctypes.CDLL(None)
            libc.prctl(38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS
    return drop


def _submission_env(mode: str, output_dir: Path,
                    extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "HF_HOME": str(HF_BUNDLE),
        "HF_HUB_OFFLINE": "1",
        "HOME": "/tmp/nanovla-home",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "NANOVLA_ACTION_DIM": ACTION_DIM,
        "NANOVLA_MODE": mode,
        "NANOVLA_OUTPUT_DIR": str(output_dir),
        "NVIDIA_DRIVER_CAPABILITIES": "compute,graphics,utility",
        "PATH": "/opt/nanovla-venv/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(PROJECT),
        "TOKENIZERS_PARALLELISM": "false",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/nanovla-inductor",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": "/tmp/nanovla-cache",
    }
    # Docker's NVIDIA runtime often exposes devices without setting this
    # variable.  Injecting an empty value would hide every GPU.
    if cuda_visible := os.environ.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    if extra:
        env.update(extra)
    return env


def _assert_isolation() -> None:
    """The image bakes these modes; the root filesystem is read-only at runtime."""
    for path in (Path("/opt/LIBERO"), Path("/opt/LIBERO-plus"), ROBOTWIN, VERIFIER_CODE, Path("/tests")):
        if path.exists() and path.stat().st_mode & 0o077:
            fail(f"verifier image isolation is misconfigured: {path} is not root-only")
    info = SHARDS_PARENT.stat()
    if info.st_uid != 0 or info.st_gid != TRAIN_UID or stat.S_IMODE(info.st_mode) != 0o750:
        fail(f"verifier image isolation is misconfigured: {SHARDS_PARENT} must be root:{TRAIN_UID} 0750")


def _scratch_sweep() -> None:
    """Remove everything a submission UID left in the scratch roots."""
    for root in SCRATCH_ROOTS:
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            try:
                owner = entry.lstat().st_uid
            except FileNotFoundError:
                continue
            if owner != 0:
                _remove(entry)


def _prepare_runtime(uid: int) -> None:
    _assert_isolation()
    _scratch_sweep()
    for path in (Path("/tmp/nanovla-home"), Path("/tmp/nanovla-cache"),
                 Path("/tmp/nanovla-inductor")):
        _reset_dir(path)
        os.chown(path, uid, uid)
    if not PRIVATE_DIR.is_dir():
        _reset_dir(PRIVATE_DIR, mode=0o700)


def _live_processes(uid: int) -> list[int]:
    try:
        own_cgroup = Path("/proc/self/cgroup").read_text()
    except (FileNotFoundError, PermissionError) as exc:
        fail(f"cannot identify verifier cgroup: {exc}")
    live = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / "cgroup").read_text() != own_cgroup:
                continue
            fields = (entry / "status").read_text().splitlines()
            uid_line = next(line for line in fields if line.startswith("Uid:"))
            state_line = next(line for line in fields if line.startswith("State:"))
        except (FileNotFoundError, PermissionError, StopIteration, ValueError):
            continue
        if int(uid_line.split()[1]) == uid and state_line.split()[1] != "Z":
            live.append(int(entry.name))
    return live


def kill_processes(uid: int) -> None:
    for _ in range(20):
        live = _live_processes(uid)
        if not live:
            return
        for pid in live:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        time.sleep(0.05)
    fail(f"submission processes survived SIGKILL: {live}")


def _terminate_group(process: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(Exception):
        process.wait(timeout=10)


def _spawn_submission(mode: str, uid: int, output_dir: Path, log_name: str,
                      extra: dict[str, str] | None = None) -> subprocess.Popen:
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    stdout = (VERIFIER_DIR / f"{log_name}.stdout.log").open("wb")
    stderr = (VERIFIER_DIR / f"{log_name}.stderr.log").open("wb")
    return subprocess.Popen(
        [sys.executable, str(STAGED_SOLUTION)],
        cwd=PROJECT,
        env=_submission_env(mode, output_dir, extra),
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
        preexec_fn=_drop_privileges(uid),
    )


# -------------------------------------------------------------------- train

def run_training(budget_s: float, *, allowance_s: float = 120.0) -> float:
    """Replay solution.py in train mode once; return its wall time.

    With NANOVLA_RESUME_DIR set, the supplied checkpoint stands in for the replay:
    the run continues the *same* policy that produced the resumed rows, which is
    the only way merging them stays honest.
    """
    validate_submission()
    if RESUME_DIR is not None and (RESUME_DIR / "ckpt.pt").is_file():
        _reset_dir(OUTPUT_DIR)
        for name in OUTPUT_FILES:
            source = RESUME_DIR / name
            if source.is_file() and source.stat().st_size:
                shutil.copy2(source, OUTPUT_DIR / name)
        os.chown(OUTPUT_DIR, TRAIN_UID, TRAIN_UID)
        for name in OUTPUT_FILES:
            if (OUTPUT_DIR / name).is_file():
                os.chown(OUTPUT_DIR / name, TRAIN_UID, TRAIN_UID)
        print(f"resuming from {RESUME_DIR}: training replay skipped", flush=True)
        return 0.0
    _prepare_runtime(TRAIN_UID)
    _reset_dir(OUTPUT_DIR)
    os.chown(OUTPUT_DIR, TRAIN_UID, TRAIN_UID)
    start = time.monotonic()
    process = _spawn_submission("train", TRAIN_UID, OUTPUT_DIR, "train")
    try:
        process.wait(timeout=budget_s + allowance_s)
    except subprocess.TimeoutExpired:
        _terminate_group(process)
        kill_processes(TRAIN_UID)
        fail(f"training replay exceeded {budget_s:.0f}s budget + {allowance_s:.0f}s allowance")
    wall = time.monotonic() - start
    kill_processes(TRAIN_UID)
    if process.returncode:
        fail(f"training replay exited with status {process.returncode}")
    if wall > budget_s + allowance_s:
        fail(f"training replay wall {wall:.1f}s exceeds the limit")
    return wall


# ------------------------------------------------------------------- freeze

def _freeze_file(source: Path, destination: Path, max_bytes: int) -> int:
    """Snapshot one submission-owned file into a root-owned, read-only copy."""
    if os.geteuid() == 0:
        try:
            os.chown(source, 0, 0, follow_symlinks=False)
        except OSError as exc:
            fail(f"could not claim {source.name}: {exc}")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source, flags)
    except OSError as exc:
        fail(f"{source.name} is not a readable regular file: {exc}")
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            fail(f"{source.name} is not a regular file")
        if before.st_nlink != 1:
            fail(f"{source.name} must have exactly one filesystem link")
        if not 1 <= before.st_size <= max_bytes:
            fail(f"{source.name} size {before.st_size} is outside 1..{max_bytes} bytes")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
        with (os.fdopen(source_fd, "rb", closefd=False) as src,
              os.fdopen(destination_fd, "wb") as dst):
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        after = os.fstat(source_fd)
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
            if getattr(before, field) != getattr(after, field):
                fail(f"{source.name} changed while the verifier froze it")
    finally:
        os.close(source_fd)
    if destination.stat().st_size != before.st_size:
        fail(f"frozen {source.name} size does not match its source")
    destination.chmod(0o444)
    return before.st_size


def freeze_output() -> Path:
    """Copy the training output into a root-owned, world-readable snapshot.

    The output directory may hold exactly ckpt.pt and, optionally, meta.json.
    Training processes are already dead, so the copy is a single immutable
    snapshot even if the written paths were adversarial.
    """
    if not OUTPUT_DIR.is_dir() or OUTPUT_DIR.is_symlink():
        fail("training replay left no output directory")
    if os.geteuid() == 0:  # claim the directory: root has no DAC override here
        os.chown(OUTPUT_DIR, 0, 0, follow_symlinks=False)
        os.chmod(OUTPUT_DIR, 0o755)
    entries = sorted(entry.name for entry in OUTPUT_DIR.iterdir())
    if "ckpt.pt" not in entries or not set(entries) <= set(OUTPUT_FILES):
        fail(f"output directory must contain ckpt.pt and at most meta.json; found {entries}")
    _reset_dir(FROZEN_DIR, mode=0o555)
    FROZEN_DIR.chmod(0o755)
    _freeze_file(OUTPUT_DIR / "ckpt.pt", FROZEN_DIR / "ckpt.pt", MAX_CHECKPOINT_BYTES)
    if "meta.json" in entries:
        _freeze_file(OUTPUT_DIR / "meta.json", FROZEN_DIR / "meta.json", MAX_META_BYTES)
        try:
            meta = json.loads((FROZEN_DIR / "meta.json").read_text(encoding="utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            fail(f"meta.json is not valid JSON: {exc}")
        if not isinstance(meta, dict):
            fail("meta.json must hold a JSON object")
    FROZEN_DIR.chmod(0o555)
    _remove(OUTPUT_DIR)
    return FROZEN_DIR


def checkpoint_tensors(frozen: Path) -> tuple[dict, int]:
    """Load ckpt.pt and flatten it to path -> tensor; return that with the element count.

    The submission writes a dict with `torch.save`; tensors may sit at the top
    level or nested in dicts, lists and tuples, and the count is every tensor
    element anywhere in the structure. `weights_only=True` already restricts
    what can be unpickled.
    """
    import torch

    def walk(node, prefix, out):
        if isinstance(node, torch.Tensor):
            out[prefix or "tensor"] = node
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else str(key), out)
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{prefix}.{index}" if prefix else str(index), out)

    try:
        payload = torch.load(frozen / "ckpt.pt", map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not payload:
            raise TypeError("ckpt.pt must be a non-empty dict")
        tensors: dict = {}
        walk(payload, "", tensors)
        if not tensors:
            raise TypeError("ckpt.pt holds no tensors")
        parameters = sum(int(value.numel()) for value in tensors.values())
        if parameters <= 0 or parameters > MAX_PARAMETERS:
            raise ValueError(f"implausible parameter count {parameters}")
    except Exception as exc:
        fail(f"ckpt.pt rejected: {type(exc).__name__}: {exc}")
    return tensors, parameters


# -------------------------------------------------------------------- serve

def evaluator_env(*, plus: bool = False, simulator: str = SIMULATOR) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "HF_HOME": str(HF_BUNDLE),
        "HF_HUB_OFFLINE": "1",
        "NANOVLA_ACTION_DIM": {"robotwin": "14"}.get(simulator, "7"),
        "NUMBA_CACHE_DIR": "/tmp/nanovla-numba",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,graphics,utility",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    })
    if simulator == "robotwin":
        # cuRobo's Warp/Triton JIT and matplotlib write under $HOME; the container is read-only
        home = Path("/tmp/nanovla-jit")
        for name in ("", "mpl", "inductor", "triton", "warp", "cache"):
            (home / name).mkdir(parents=True, exist_ok=True)
        env.update({
            "HOME": str(home),
            "MPLCONFIGDIR": str(home / "mpl"),
            "NANOVLA_ROBOTWIN_ROOT": str(ROBOTWIN),
            "PYTHONPATH": f"{VERIFIER_CODE}:{PROJECT}:{ROBOTWIN}",
            "TORCHINDUCTOR_CACHE_DIR": str(home / "inductor"),
            "TRITON_CACHE_DIR": str(home / "triton"),
            "WARP_CACHE_PATH": str(home / "warp"),
            "XDG_CACHE_HOME": str(home / "cache"),
        })
    elif plus:
        env.update({
            "LIBERO_CONFIG_PATH": "/opt/nanovla-config/libero_plus",
            "MUJOCO_GL": "egl",
            "NANOVLA_LIBERO_PLUS_ROOT": "/opt/LIBERO-plus/libero/libero",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONPATH": f"/opt/LIBERO-plus:{VERIFIER_CODE}:{PROJECT}",
        })
    else:
        env.update({
            "LIBERO_CONFIG_PATH": "/opt/nanovla-config/libero",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONPATH": f"{VERIFIER_CODE}:{PROJECT}:/opt/LIBERO",
        })
    return env


def _override(name: str, value, cast):
    """An operator override for one rollout resource, applied over the subtask's value.

    Unset in every shipped task.toml, so the scored configuration is the default;
    a host with different GPUs can resize a rollout without rebuilding the image.
    """
    raw = os.environ.get(name)
    if not raw:
        return value
    return cast(raw)


def run_rollout(name: str, socket_path: Path, args: list[str], *, timeout_s: float,
                plus: bool = False, workers: int, gpus: int, simulator: str = SIMULATOR) -> Path:
    """Run the simulator's rollout script against the policy socket; return its rows path."""
    workers = _override("NANOVLA_ROLLOUT_WORKERS", workers, int)
    gpus = _override("NANOVLA_ROLLOUT_GPUS", gpus, int)
    timeout_s = _override("NANOVLA_ROLLOUT_TIMEOUT_S", timeout_s, float)
    _reset_dir(Path("/tmp/nanovla-numba"), mode=0o700)
    # The driver resumes from <name>.jsonl.inc under /logs/verifier. Anything already
    # there was not written by this verifier: clear the pair first. A genuine resume
    # point arrives through RESUME_DIR, a separate read-only mount no agent ever sees.
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    output = VERIFIER_DIR / f"{name}.jsonl"
    target = Path(str(output) + ".inc")
    for stale in (output, target):
        if stale.is_dir() and not stale.is_symlink():
            shutil.rmtree(stale)
        else:
            stale.unlink(missing_ok=True)
    if RESUME_DIR is not None:
        carried = RESUME_DIR / f"{name}.jsonl.inc"
        if carried.is_file():
            shutil.copy2(carried, target)
            print(f"resuming {name}: {sum(1 for _ in carried.open())} episodes carried over", flush=True)
    script = "rollout_robotwin.py" if simulator == "robotwin" else "rollout.py"
    command = [sys.executable, str(VERIFIER_CODE / script),
               "--remote-socket", str(socket_path), "--out", str(output),
               "--resume", str(output) + ".inc", "--workers", str(workers),
               "--gpus", str(gpus), "--record", str(VERIFIER_DIR / "media"), *args]
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    with ((VERIFIER_DIR / f"{name}.stdout.log").open("wb") as stdout,
          (VERIFIER_DIR / f"{name}.stderr.log").open("wb") as stderr):
        process = subprocess.Popen(command, cwd=VERIFIER_CODE, env=evaluator_env(plus=plus, simulator=simulator),
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _terminate_group(process)
            fail(f"{name} exceeded {timeout_s:.0f}s evaluator timeout")
    if process.returncode:
        fail(f"{name} exited with status {process.returncode}")
    return output


def serve_and_evaluate(evaluate: Callable[[Path, float], object], *, hard_cap_s: float,
                       startup_s: float = 300.0):
    """Start solution.py in serve mode as the serving UID and run `evaluate` against it.

    `evaluate(socket_path, remaining_s)` runs as root while the server is alive;
    its return value is passed through. The server is killed afterwards.
    """
    hard_cap_s = _override("NANOVLA_ROLLOUT_HARD_CAP_S", hard_cap_s, float)
    if not (FROZEN_DIR / "ckpt.pt").is_file():
        fail("serve phase needs a frozen checkpoint")
    kill_processes(TRAIN_UID)
    _prepare_runtime(SERVE_UID)
    _reset_dir(IPC_DIR, mode=0o755)  # root has no DAC override: it must traverse to connect
    os.chown(IPC_DIR, SERVE_UID, SERVE_UID)
    socket_path = IPC_DIR / "policy.sock"
    start = time.monotonic()
    server = _spawn_submission("serve", SERVE_UID, FROZEN_DIR, "serve",
                               {"NANOVLA_SOCKET": str(socket_path)})
    try:
        deadline = start + min(startup_s, hard_cap_s)
        while True:
            if server.poll() is not None:
                fail(f"policy server exited with status {server.returncode}")
            with contextlib.suppress(FileNotFoundError):
                if stat.S_ISSOCK(socket_path.lstat().st_mode):
                    break
            if time.monotonic() >= deadline:
                fail(f"policy server did not create NANOVLA_SOCKET within {startup_s:.0f}s")
            time.sleep(0.1)
        remaining = hard_cap_s - (time.monotonic() - start)
        if remaining <= 0:
            fail("serve phase exhausted its wall-clock cap during startup")
        return evaluate(socket_path, remaining)
    finally:
        _terminate_group(server)
        kill_processes(SERVE_UID)


# ------------------------------------------------------------------- scoring

def standard_expected(suite: str, tasks: range | list[int], episodes: int) -> set[tuple]:
    return {("standard", suite, int(task), episode) for task in tasks for episode in range(episodes)}


def plus_expected(manifest: list[dict]) -> set[tuple]:
    return {("plus", entry["bddl"], int(entry["init_idx"])) for entry in manifest}


def robotwin_expected(book: dict, per_task: int = 0) -> set[tuple]:
    """Keys of a RoboTwin episode book: {"tasks": [...], "episodes": {task: [{seed, ...}, ...]}}."""
    keys = set()
    for task in book["tasks"]:
        entries = book["episodes"][task]
        for entry in entries[:per_task] if per_task else entries:
            keys.add(("robotwin", task, int(entry["seed"])))
    return keys


def _row_key(row: dict) -> tuple:
    if row.get("kind") == "plus":
        return ("plus", str(row["bddl"]), int(row["init_idx"]))
    if row.get("kind") == "robotwin":
        return ("robotwin", str(row["task"]), int(row["seed"]))
    return ("standard", str(row["suite"]), int(row["task"]), int(row["episode"]))


def score_rows(path: Path, expected: set[tuple]) -> tuple[float | None, int]:
    """Success rate over `expected` with missing episodes as failures."""
    rows: dict[tuple, bool] = {}
    for candidate in (path, Path(str(path) + ".inc")):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
                key = _row_key(row)
                if type(row["success"]) is not bool:
                    continue
                if key in expected:
                    rows[key] = row["success"]
            except Exception:
                continue
    if not expected:
        return None, 0
    completed = len(rows.keys() & expected)
    return sum(rows.get(key, False) for key in expected) / len(expected), completed


def per_task_rates(path: Path, expected: set[tuple]) -> dict[str, float]:
    rates: dict[str, list[bool]] = {}
    for candidate in (path, Path(str(path) + ".inc")):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
                key = _row_key(row)
            except Exception:
                continue
            if key in expected and type(row.get("success")) is bool:
                group = str(row["base"]) if key[0] == "plus" else str(row["task"])
                rates.setdefault(group, []).append(row["success"])
    return {group: sum(values) / len(values) for group, values in sorted(rates.items())}


def write_result(reward: float, report: dict, reason: str | None = None) -> None:
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    reward = min(1.0, max(0.0, float(reward)))
    if reason:
        report = {**report, "failure": reason}
    (VERIFIER_DIR / "reward.json").write_text(
        json.dumps({"reward": round(reward, 6)}, indent=2) + "\n"
    )
    (VERIFIER_DIR / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    print(json.dumps({"reward": reward, **report}, indent=2, default=str))


def guarded_main(score) -> None:
    try:
        reward, report = score()
        write_result(reward, report)
    except ScoreFailure as exc:
        write_result(0.0, {}, str(exc))
    except Exception as exc:
        write_result(0.0, {"scoring_error": type(exc).__name__}, str(exc))
        raise
