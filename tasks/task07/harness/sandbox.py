"""Isolated CLOSED-LOOP execution of untrusted task07 policies.

Fresh subprocess, submission-only sys.path, scrubbed env, uid drop to
nobody under root with process/CPU/memory limits, /tests chmod 700 by
test.sh. The child gets a private writable home; on close every uid-65534
process and IPC object is removed, so nothing outlives its episode. The
loop is interactive: the parent steps physics and exchanges one message per
control tick.

Wire protocol (pipes, parent <-> child):
  parent -> child : 8-byte big-endian length + npz payload
                    (kind 0 = reset {spec_json, seed}, 1 = act {obs...},
                     2 = shutdown)
  child  -> parent: exactly 65 bytes — 1 tag byte (0 ok / 1 fault) +
                    8 float64 ctrl values
The parent never unpickles agent bytes: replies are fixed-size raw floats,
validated for shape and finiteness upstream. Timeouts (startup and per-act
hang caps) are enforced on the PARENT's clock; a dead child yields None
forever and the episode runs on with held ctrl.

The child's stdout is re-pointed at stderr before any agent code runs, so
print() cannot corrupt the protocol stream.
"""
from __future__ import annotations

import ctypes
import json
import os
import resource
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from io import BytesIO

import numpy as np

from . import scene, spec

STARTUP_TIMEOUT_S = 120.0    # spawn + imports (torch headroom) + reset()
POLICY_UID = 65534
# Threads count toward NPROC, and the count is per uid across the host:
# a fork-bomb guard well above torch's thread pools, not a tight budget.
POLICY_LIMITS = (
    (resource.RLIMIT_NPROC, 4096),
    (resource.RLIMIT_CPU, int(2 * spec.EPISODE_WALL_BUDGET_S)),
    (resource.RLIMIT_AS, 16 * 1024**3),   # address space; the container caps RSS
    (resource.RLIMIT_FSIZE, 256 * 1024**2),
    (resource.RLIMIT_MSGQUEUE, 0),
    (resource.RLIMIT_CORE, 0),
)
_LIBC = ctypes.CDLL(None, use_errno=True)
KIND_RESET, KIND_ACT, KIND_END = 0, 1, 2
REPLY_LEN = 1 + 8 * spec.N_CTRL

_RUNNER = r'''
import ctypes, errno, importlib.util, json, os, struct, sys
import numpy as np

# protect the protocol stream: agent prints go to stderr
_out = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr
_in = os.fdopen(os.dup(0), "rb")

N_CTRL = %(n_ctrl)d
REPLY_LEN = 1 + 8 * N_CTRL
_LIBC = ctypes.CDLL(None, use_errno=True)


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8),
                ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(_SockFilter))]


def _block_subprocesses(strict):
    machine = os.uname().machine if hasattr(os, "uname") else ""
    spec = {
        "x86_64": (0xC000003E, 56, 435, (57, 58, 59, 322)),
        "aarch64": (0xC00000B7, 220, 435, (221, 281)),
        "arm64": (0xC00000B7, 220, 435, (221, 281)),
    }.get(machine)
    if spec is None:
        if strict:
            raise RuntimeError("unsupported seccomp architecture: " + machine)
        return
    arch, nr_clone, nr_clone3, deny_nrs = spec
    raw = [
        (0x20, 0, 0, 4),
        (0x15, 1, 0, arch),
        (0x06, 0, 0, 0x00050000 | errno.EPERM),
        (0x20, 0, 0, 0),
        (0x35, 0, 1, 0x40000000),
        (0x06, 0, 0, 0x00050000 | errno.EPERM),
        (0x15, 0, 1, nr_clone3),
        (0x06, 0, 0, 0x00050000 | errno.ENOSYS),
    ]
    for nr in deny_nrs:
        raw += [(0x15, 0, 1, nr), (0x06, 0, 0, 0x00050000 | errno.EAGAIN)]
    raw += [
        (0x15, 1, 0, nr_clone),
        (0x06, 0, 0, 0x7FFF0000),
        (0x20, 0, 0, 16),
        (0x45, 0, 1, 0x00010000),
        (0x06, 0, 0, 0x7FFF0000),
        (0x06, 0, 0, 0x00050000 | errno.EAGAIN),
    ]
    arr = (_SockFilter * len(raw))(*(_SockFilter(*ins) for ins in raw))
    prog = _SockFprog(len(raw), arr)
    if _LIBC.prctl(38, 1, 0, 0, 0):
        if strict:
            raise OSError(ctypes.get_errno(), "no_new_privs")
        return
    if _LIBC.prctl(22, 2, ctypes.byref(prog), 0, 0) and strict:
        raise OSError(ctypes.get_errno(), "seccomp")


def _read_msg():
    head = _in.read(8)
    if len(head) < 8:
        return None
    n = struct.unpack(">Q", head)[0]
    buf = b""
    while len(buf) < n:
        chunk = _in.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    from io import BytesIO
    return np.load(BytesIO(buf), allow_pickle=False)


def _reply(tag, ctrl=None):
    vals = np.zeros(N_CTRL) if ctrl is None else \
        np.asarray(ctrl, dtype=np.float64).reshape(-1)[:N_CTRL]
    if vals.shape[0] < N_CTRL:
        vals = np.concatenate([vals, np.zeros(N_CTRL - vals.shape[0])])
    _out.write(bytes([tag]) + vals.astype(">f8").tobytes())
    _out.flush()


def main():
    _block_subprocesses(strict=(os.geteuid() == 65534))
    sub_dir = sys.argv[1]
    sys.path.insert(0, sub_dir)
    mspec = importlib.util.spec_from_file_location(
        "policy", os.path.join(sub_dir, "policy.py"))
    mod = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(mod)
    policy = mod.make_policy()
    _reply(0)
    faults = 0

    while True:
        msg = _read_msg()
        if msg is None:
            return
        kind = int(msg["_kind"])
        if kind == 2:
            return
        if kind == 0:
            try:
                cell = json.loads(bytes(msg["spec_json"]).decode())
                policy.reset(cell, int(msg["seed"]))
                _reply(0)
            except Exception:
                import traceback; traceback.print_exc()
                _reply(1)
            continue
        obs = {}
        for k in msg.files:
            if k == "_kind":
                continue
            v = msg[k]
            obs[k] = float(v) if v.ndim == 0 else v
        try:
            ctrl = np.asarray(policy.act(obs), dtype=np.float64).reshape(-1)
            if ctrl.shape != (N_CTRL,):
                raise ValueError(f"ctrl shape {ctrl.shape}")
            _reply(0, ctrl)
        except Exception:
            faults += 1
            if faults <= 20:
                import traceback; traceback.print_exc()
            _reply(1)


main()
''' % {"n_ctrl": spec.N_CTRL}


def _drop_privileges():
    """Pre-exec: become nobody so /tests (chmod 700) is unreadable."""
    for name, value in POLICY_LIMITS:
        resource.setrlimit(name, (value, value))
    # Out of memory, the kernel kills the policy, not the scorer; only root
    # can lower this again.
    with open("/proc/self/oom_score_adj", "w") as stream:
        stream.write("1000")
    os.setgroups([])
    os.setgid(POLICY_UID)
    os.setuid(POLICY_UID)
    if _LIBC.prctl(38, 1, 0, 0, 0):  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "no_new_privs")


def _reap_policy_uid() -> None:
    """Kill every policy-uid process and drop its SysV IPC objects. Only
    policies run as this uid, so this also catches escaped descendants. Runs
    as that uid: container root cannot remove another owner's IPC."""
    pid = os.fork()
    if pid:
        os.waitpid(pid, 0)
        return
    try:
        os.setgroups([])
        os.setgid(POLICY_UID)
        os.setuid(POLICY_UID)
        os.kill(-1, signal.SIGKILL)   # every process of this uid but us
        for kind, remove in (("shm", lambda i: _LIBC.shmctl(i, 0, None)),
                             ("msg", lambda i: _LIBC.msgctl(i, 0, None)),
                             ("sem", lambda i: _LIBC.semctl(i, 0, 0))):
            try:
                with open(f"/proc/sysvipc/{kind}") as stream:
                    header, *rows = [line.split() for line in stream]
            except FileNotFoundError:
                continue
            for row in rows:
                if int(row[header.index("uid")]) == POLICY_UID:
                    remove(int(row[1]))   # IPC_RMID
    finally:
        os._exit(0)


def _world_readable(path: str) -> None:
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o755)
        for f in files:
            p = os.path.join(root, f)
            os.chmod(p, (os.stat(p).st_mode | 0o644) & ~0o022)
    os.chmod(path, 0o755)


def _pack(kind: int, arrays: dict) -> bytes:
    buf = BytesIO()
    np.savez(buf, _kind=np.array(kind), **arrays)
    payload = buf.getvalue()
    return struct.pack(">Q", len(payload)) + payload


class PolicyLoadError(RuntimeError):
    """The submitted module or factory could not be loaded."""


class PolicyProcess:
    """Sandboxed submission implementing the runtime Policy interface.

    act() returns the child's ctrl array, or None on any fault (bad reply,
    hang, dead child) — the runtime holds the last valid ctrl and counts it.
    """

    def __init__(self, submission_dir: str,
                 act_timeout_s: float = spec.ACT_HANG_CAP_S,
                 startup_timeout_s: float = STARTUP_TIMEOUT_S):
        self.submission_dir = os.path.abspath(submission_dir)
        self.act_timeout_s = act_timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.dead: str | None = None
        self.loaded = False
        self.load_wall_s = 0.0
        # Parent-observed submission time. The episode budget deliberately
        # excludes MuJoCo stepping, sensor rendering, and verifier video.
        self.reset_wall_s = 0.0
        self.act_wall_s = 0.0
        self.n_acts = 0
        self.work = tempfile.mkdtemp(prefix="task07_")
        os.chmod(self.work, 0o755)
        self.sandboxed = os.geteuid() == 0

        if not os.path.exists(os.path.join(self.submission_dir,
                                           "policy.py")):
            self.dead = "policy.py missing"
            self.proc = None
            return
        staged = os.path.join(self.work, "policy")
        shutil.copytree(self.submission_dir, staged, symlinks=True)
        for root, dirs, files in os.walk(staged):
            if any(os.path.islink(os.path.join(root, name)) for name in dirs + files):
                self.dead = "policy package must not contain symbolic links"
                self.proc = None
                return
        self.submission_dir = staged
        _world_readable(staged)

        self.cell_mjb = os.path.join(self.work, "cell.mjb")
        scene.save_cell_mjb(self.cell_mjb)
        os.chmod(self.cell_mjb, 0o644)
        runner = os.path.join(self.work, "runner.py")
        with open(runner, "w") as f:
            f.write(_RUNNER)
        os.chmod(runner, 0o644)
        self._stderr_path = os.path.join(self.work, "stderr.log")
        self._stderr_f = open(self._stderr_path, "wb")
        os.chmod(self._stderr_path, 0o666)

        home = os.path.join(self.work, "home")
        os.mkdir(home, 0o700)
        if self.sandboxed:
            os.chown(home, POLICY_UID, POLICY_UID)
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": home, "TMPDIR": home,
            "MPLCONFIGDIR": home, "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MUJOCO_GL": os.environ.get("MUJOCO_GL", "osmesa"),
        }
        preexec = _drop_privileges if self.sandboxed else None
        self.proc = subprocess.Popen(
            # No user site or script-dir imports; isolated mode (-I) would drop PYTHONHASHSEED.
            [sys.executable, "-s", "-P", runner, self.submission_dir],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr_f, env=env, cwd=home,
            preexec_fn=preexec, start_new_session=True)

    def check_loaded(self) -> None:
        """Wait for module import and make_policy(), before starting physics."""
        if self.loaded:
            return
        t0 = time.monotonic()
        try:
            if not self.dead and self._read_reply(self.startup_timeout_s) is not None:
                self.loaded = True
                return
        finally:
            self.load_wall_s += time.monotonic() - t0
        raise PolicyLoadError(f"policy load failed: {self.dead}\n{self.stderr_tail()}")

    # ------------------------------------------------------------- protocol
    def _send(self, kind: int, arrays: dict) -> bool:
        try:
            self.proc.stdin.write(_pack(kind, arrays))
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            self._kill("child pipe closed")
            return False

    def _read_reply(self, timeout_s: float) -> np.ndarray | None:
        fd = self.proc.stdout.fileno()
        deadline = time.monotonic() + timeout_s
        buf = b""
        while len(buf) < REPLY_LEN:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill(f"reply timeout after {timeout_s:.0f}s")
                return None
            r, _, _ = select.select([fd], [], [], min(remaining, 1.0))
            if not r:
                continue
            chunk = os.read(fd, REPLY_LEN - len(buf))
            if not chunk:
                self._kill("child exited")
                return None
            buf += chunk
        if buf[0] != 0:
            return None      # child-reported fault (episode continues)
        return np.frombuffer(buf[1:], dtype=">f8").astype(np.float64)

    def _kill(self, reason: str) -> None:
        if self.dead is None:
            self.dead = reason
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    # ------------------------------------------------------------ interface
    def reset(self, cell_spec: dict, seed: int) -> None:
        if self.dead:
            return
        self.check_loaded()
        # Charge preflight import time to the episode's first reset.
        self.reset_wall_s += self.load_wall_s
        self.load_wall_s = 0.0
        cell = dict(cell_spec)
        cell["model_file"] = self.cell_mjb
        spec_json = json.dumps(cell).encode()
        t0 = time.monotonic()
        try:
            if not self._send(KIND_RESET, {
                    "spec_json": np.frombuffer(spec_json, dtype=np.uint8),
                    "seed": np.array(int(seed))}):
                return
            if self._read_reply(self.startup_timeout_s) is None:
                self._kill("policy reset failed")
        finally:
            self.reset_wall_s += time.monotonic() - t0

    def act(self, obs: dict):
        if self.dead:
            return None
        arrays = {}
        for k, v in obs.items():
            arrays[k] = np.asarray(v)
        t0 = time.monotonic()
        try:
            if not self._send(KIND_ACT, arrays):
                return None
            return self._read_reply(self.act_timeout_s)
        finally:
            self.act_wall_s += time.monotonic() - t0
            self.n_acts += 1

    @property
    def policy_wall_s(self) -> float:
        """Cumulative parent-observed time spent waiting on submission code."""
        return self.reset_wall_s + self.act_wall_s

    def stderr_tail(self, n: int = 2000) -> str:
        try:
            with open(self._stderr_path, "rb") as f:
                return f.read()[-n:].decode(errors="replace")
        except (OSError, AttributeError):
            return ""

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self._send(KIND_END, {})
                self.proc.wait(timeout=5)
            except Exception:
                pass
            self._kill("closed")
        if self.proc is not None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if self.sandboxed:
                _reap_policy_uid()
        if getattr(self, "_stderr_f", None) is not None:
            self._stderr_f.close()
        shutil.rmtree(self.work, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
