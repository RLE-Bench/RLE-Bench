"""Pipe worker for trim.py: one JSON request in, one JSON result out.

Copied next to the submission and run as a standalone script; never imports
the verifier harness. The parent passes the nominal model as model.mjb.
"""
import ctypes
import errno
import importlib.util
import json
import os
import random
import resource
import struct
import sys

import mujoco
import numpy as np

_LIBC = ctypes.CDLL(None, use_errno=True)


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8),
                ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(_SockFilter))]


def _block_subprocesses(strict: bool) -> None:
    """Allow threads (CLONE_THREAD) while blocking fork/vfork/clone/execve."""
    machine = os.uname().machine if hasattr(os, "uname") else ""
    spec = {
        "x86_64": (0xC000003E, 56, 435, (57, 58, 59, 322)),
        "aarch64": (0xC00000B7, 220, 435, (221, 281)),
        "arm64": (0xC00000B7, 220, 435, (221, 281)),
    }.get(machine)
    if spec is None:
        if strict:
            raise RuntimeError(f"unsupported seccomp architecture: {machine}")
        return
    arch, nr_clone, nr_clone3, deny_nrs = spec
    raw = [
        (0x20, 0, 0, 4),                          # arch
        (0x15, 1, 0, arch),
        (0x06, 0, 0, 0x00050000 | errno.EPERM),
        (0x20, 0, 0, 0),                          # syscall nr
        (0x35, 0, 1, 0x40000000),                 # block x32 ABI bit
        (0x06, 0, 0, 0x00050000 | errno.EPERM),
        (0x15, 0, 1, nr_clone3),                  # clone3 -> ENOSYS fallback
        (0x06, 0, 0, 0x00050000 | errno.ENOSYS),
    ]
    for nr in deny_nrs:
        raw += [(0x15, 0, 1, nr), (0x06, 0, 0, 0x00050000 | errno.EAGAIN)]
    raw += [
        (0x15, 1, 0, nr_clone),                   # allow non-clone syscalls
        (0x06, 0, 0, 0x7FFF0000),
        (0x20, 0, 0, 16),                         # clone flags (args[0] low)
        (0x45, 0, 1, 0x00010000),                 # CLONE_THREAD
        (0x06, 0, 0, 0x7FFF0000),
        (0x06, 0, 0, 0x00050000 | errno.EAGAIN),
    ]
    arr = (_SockFilter * len(raw))(*(_SockFilter(*ins) for ins in raw))
    prog = _SockFprog(len(raw), arr)
    if _LIBC.prctl(38, 1, 0, 0, 0):  # PR_SET_NO_NEW_PRIVS
        if strict:
            raise OSError(ctypes.get_errno(), "no_new_privs")
        return
    if _LIBC.prctl(22, 2, ctypes.byref(prog), 0, 0) and strict:  # PR_SET_SECCOMP
        raise OSError(ctypes.get_errno(), "seccomp")


def _limit(cpu_seconds: int) -> None:
    sandboxed = os.geteuid() == 0
    if sandboxed:
        for name, value in ((resource.RLIMIT_CORE, 0),
                            (resource.RLIMIT_NPROC, 256),
                            (resource.RLIMIT_CPU, cpu_seconds),
                            (resource.RLIMIT_AS, 4 * 1024**3),
                            (resource.RLIMIT_FSIZE, 64 * 1024**2),
                            (resource.RLIMIT_MSGQUEUE, 0),
                            (resource.RLIMIT_NOFILE, 128)):
            resource.setrlimit(name, (value, value))
        with open("/proc/self/oom_score_adj", "w") as stream:
            stream.write("1000")
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    _block_subprocesses(strict=sandboxed)


def _trim_rows(trim, qmat, nv: int) -> list:
    """Sample a trim callable on a query matrix; bad rows degrade to zeros."""
    rows = []
    for qq in qmat:
        try:
            t = np.asarray(trim(np.asarray(qq, dtype=float).copy()),
                           dtype=float).reshape(-1)
            ok = t.shape[0] == nv and np.all(np.isfinite(t))
            rows.append(t.tolist() if ok else [0.0] * nv)
        except Exception:
            rows.append([0.0] * nv)
    return rows


def _run(mod, model_file: str, request: dict) -> dict:
    """Jobs are anonymous lists; the parent alone maps them to instances."""
    load = lambda: mujoco.MjModel.from_binary_path(model_file)
    result = {}
    if "plan_probe" in request:
        plans = []
        for job in request["plan_probe"]:
            try:
                model = load()
                sample = tuple(np.asarray(q, dtype=float) for q in job["sample"])
                plan = np.asarray(mod.plan_probe(
                    model, sample, np.asarray(job["lower"], dtype=float),
                    np.asarray(job["upper"], dtype=float)), dtype=float)
                if (plan.shape != (request["chosen_poses"], model.nv)
                        or not np.all(np.isfinite(plan))):
                    raise ValueError("invalid probe plan shape or values")
                plans.append({"configs": plan.tolist()})
            except Exception as exc:
                plans.append({"error": str(exc)[:500]})
        result["plan_probe"] = plans
    if "ctrim" in request:
        model = load()
        trim = mod.make_trim(model)
        result["ctrim"] = [_trim_rows(trim, q, model.nv) for q in request["ctrim"]]
    if "cadapt" in request:
        model = load()
        out = []
        for job in request["cadapt"]:
            try:
                probe = [(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
                         for a, b in job["probe"]]
                trim = mod.make_trim_adapted(model, mod.adapt(model, probe))
                out.append([_trim_rows(trim, q, model.nv) for q in job["qmats"]])
            except Exception:
                out.append(None)
        result["cadapt"] = out
    return result


def main():
    work, cpu_seconds = sys.argv[1], int(sys.argv[2])
    output = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    size = struct.unpack(">Q", sys.stdin.buffer.read(8))[0]
    request = json.loads(sys.stdin.buffer.read(size))
    os.chdir(os.path.join(work, "home"))
    _limit(cpu_seconds)
    random.seed(0)
    np.random.seed(0)
    submission = os.path.join(work, "submission")
    sys.path[:0] = [submission, os.path.join(work, "public")]
    try:
        spec = importlib.util.spec_from_file_location(
            "agent_trim", os.path.join(submission, "trim.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        reply = {"ok": _run(mod, os.path.join(work, "model.mjb"), request)}
    except Exception as exc:  # missing file, un-importable module, etc.
        reply = {"err": repr(exc)[:500]}
    payload = json.dumps(reply, allow_nan=False).encode()
    output.write(struct.pack(">Q", len(payload)) + payload)


if __name__ == "__main__":
    main()
