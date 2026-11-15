"""Run the untrusted trim.py in a fresh, unprivileged interpreter and return
only plain arrays.

The child is a new `python -s -P` process (trim_worker.py) that never imports
the verifier harness. It sees a private copy of the submission, the public
helper package, and the nominal model as an .mjb file. Under root it drops to
uid 65534 with no_new_privs, no forks, and CPU/memory/file limits; the whole
process group is killed when the call ends. Jobs cross the pipe as anonymous
lists, so instance identities stay in this process.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
from pathlib import Path
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

import mujoco
import numpy as np

from . import spec as gspec
from . import trim_worker
from .probe_protocol import CHOSEN_POSES
from .public_package import write_package
from .scenarios import compose_lead

DEFAULT_TIMEOUT_S = 120.0
MAX_REPLY_BYTES = 256 * 1024**2
PRIVATE_DIRS = ("/tests", "/logs/verifier")
TRIM_UID = 65534
_LIBC = ctypes.CDLL(None, use_errno=True)


def _check_private_dirs() -> None:
    """Fail closed if the verifier runs without its permission boundary."""
    for directory in PRIVATE_DIRS:
        if not os.path.isdir(directory) or os.stat(directory).st_mode & 0o077:
            raise RuntimeError(f"private verifier directory is not protected: {directory}")


def _reap_trim_uid() -> None:
    """Kill every uid-65534 process and drop its SysV IPC objects."""
    pid = os.fork()
    if pid:
        os.waitpid(pid, 0)
        return
    try:
        os.setgroups([])
        os.setgid(TRIM_UID)
        os.setuid(TRIM_UID)
        os.kill(-1, signal.SIGKILL)
        for kind, remove in (("shm", lambda i: _LIBC.shmctl(i, 0, None)),
                             ("msg", lambda i: _LIBC.msgctl(i, 0, None)),
                             ("sem", lambda i: _LIBC.semctl(i, 0, 0))):
            try:
                with open(f"/proc/sysvipc/{kind}") as stream:
                    header, *rows = [line.split() for line in stream]
            except FileNotFoundError:
                continue
            for row in rows:
                if int(row[header.index("uid")]) == TRIM_UID:
                    remove(int(row[1]))
    finally:
        os._exit(0)


def _stage(work: Path, submission_dir: str, lead_xml: str,
           sandboxed: bool) -> Path:
    staged = work / "submission"
    shutil.copytree(submission_dir, staged, symlinks=True)
    for root, dirs, files in os.walk(staged):
        if any(os.path.islink(os.path.join(root, name)) for name in dirs + files):
            raise ValueError("submission must not contain symbolic links")
    write_package(str(work / "public" / "harness"), gspec.VARIANT_NAME)
    shutil.copyfile(trim_worker.__file__, work / "worker.py")
    model, _ = compose_lead(lead_xml, seed=0, perturb=False)
    mujoco.mj_saveModel(model, str(work / "model.mjb"))
    os.chmod(work, 0o755)
    for root, dirs, files in os.walk(work):
        for name in dirs:
            os.chmod(os.path.join(root, name), 0o755)
        for name in files:
            p = os.path.join(root, name)
            os.chmod(p, (os.stat(p).st_mode | 0o644) & ~0o022)
    home = work / "home"
    home.mkdir(mode=0o700)
    if sandboxed:
        os.chown(home, TRIM_UID, TRIM_UID)
    return home


def _request(jobs: dict) -> tuple[dict, dict]:
    """Strip instance keys: the child gets ordered lists, the parent keeps
    the order to map results back."""
    request: dict = {"chosen_poses": CHOSEN_POSES}
    order: dict = {}
    if "plan_probe" in jobs:
        order["plan_probe"] = list(jobs["plan_probe"])
        request["plan_probe"] = [
            dict(sample=[np.asarray(q, dtype=float).tolist() for q in job["sample"]],
                 lower=np.asarray(job["lower"], dtype=float).tolist(),
                 upper=np.asarray(job["upper"], dtype=float).tolist())
            for job in jobs["plan_probe"].values()]
    if "ctrim" in jobs:
        order["ctrim"] = list(jobs["ctrim"])
        request["ctrim"] = [np.asarray(q, dtype=float).tolist()
                            for q in jobs["ctrim"].values()]
    if "cadapt" in jobs:
        order["cadapt"] = [(key, list(job["qmats"])) for key, job in jobs["cadapt"].items()]
        request["cadapt"] = [
            dict(probe=[[np.asarray(a, dtype=float).tolist(),
                         np.asarray(b, dtype=float).tolist()] for a, b in job["probe"]],
                 qmats=[np.asarray(q, dtype=float).tolist() for q in job["qmats"].values()])
            for job in jobs["cadapt"].values()]
    return request, order


def _restore(result: dict, order: dict) -> dict:
    out: dict = {}
    if "plan_probe" in order:
        out["plan_probe"] = {}
        for key, plan in zip(order["plan_probe"], result["plan_probe"], strict=True):
            out["plan_probe"][key] = (
                {"configs": np.asarray(plan["configs"], dtype=float)}
                if "configs" in plan else {"error": str(plan.get("error"))})
    if "ctrim" in order:
        out["ctrim"] = {key: np.asarray(rows, dtype=float) for key, rows in
                        zip(order["ctrim"], result["ctrim"], strict=True)}
    if "cadapt" in order:
        out["cadapt"] = {}
        for (key, names), mats in zip(order["cadapt"], result["cadapt"], strict=True):
            out["cadapt"][key] = None if mats is None else {
                name: np.asarray(rows, dtype=float)
                for name, rows in zip(names, mats, strict=True)}
    return out


def _exchange(proc, payload: bytes, timeout: float) -> bytes:
    outgoing = memoryview(struct.pack(">Q", len(payload)) + payload)
    incoming = bytearray()
    expected = None
    deadline = time.monotonic() + timeout
    os.set_blocking(proc.stdin.fileno(), False)
    os.set_blocking(proc.stdout.fileno(), False)
    while expected is None or len(incoming) < expected:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("trim.py timed out")
        reads, writes, _ = select.select(
            [proc.stdout], [proc.stdin] if outgoing else [], [], remaining)
        if writes:
            outgoing = outgoing[os.write(proc.stdin.fileno(), outgoing):]
            if not outgoing:
                proc.stdin.close()
        if reads:
            chunk = os.read(proc.stdout.fileno(), 1 << 20)
            if not chunk:
                raise EOFError("trim worker exited")
            incoming.extend(chunk)
            if expected is None and len(incoming) >= 8:
                expected = 8 + struct.unpack(">Q", incoming[:8])[0]
                if expected > MAX_REPLY_BYTES:
                    raise ValueError("oversized trim worker reply")
    return bytes(incoming[8:expected])


def run_jobs(submission_dir: str, lead_xml: str | None, jobs: dict,
             timeout: float = DEFAULT_TIMEOUT_S) -> dict | None:
    """Execute `jobs` in an isolated subprocess. Returns the result dict, or
    None if the child errored, timed out, or crashed."""
    if not jobs:
        return {}
    sandboxed = os.geteuid() == 0
    if sandboxed:
        _check_private_dirs()
    work = Path(tempfile.mkdtemp(prefix="task09-trim-"))
    proc = None
    try:
        try:
            home = _stage(work, submission_dir, lead_xml, sandboxed)
        except Exception:
            return None
        request, order = _request(jobs)
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home),
               "TMPDIR": str(home), "MPLCONFIGDIR": str(home),
               "PYTHONHASHSEED": "0", "OPENBLAS_NUM_THREADS": "1",
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "MUJOCO_GL": "osmesa"}
        proc = subprocess.Popen(
            [sys.executable, "-s", "-P", str(work / "worker.py"), str(work),
             str(math.ceil(timeout))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, cwd="/",
            start_new_session=True)
        try:
            raw = _exchange(proc, json.dumps(request).encode(), timeout)
            reply = json.loads(raw)
            return _restore(reply["ok"], order) if "ok" in reply else None
        except Exception:  # timeout, crash, or a malformed reply
            return None
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            if sandboxed:
                _reap_trim_uid()
            for stream in (proc.stdin, proc.stdout):
                if not stream.closed:
                    stream.close()
        shutil.rmtree(work, ignore_errors=True)
