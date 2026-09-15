"""Isolated execution of an untrusted single-file controller."""

from __future__ import annotations

import json
import os
import select
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
from io import BytesIO

import numpy as np

from . import model as model_lib
from . import spec

REPLY_BYTES = 1 + 8 * spec.N_JOINTS
RESET, ACT, STOP = 0, 1, 2

_RUNNER = r'''
import ctypes, importlib.util, json, os, resource, struct, sys
from io import BytesIO
import mujoco
import numpy as np

wire_out = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr
wire_in = os.fdopen(os.dup(0), "rb")
if os.geteuid() == 0:
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
if ctypes.CDLL(None).prctl(38, 1, 0, 0, 0):
    raise RuntimeError("could not set no-new-privileges")
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

def read_message():
    header = wire_in.read(8)
    if len(header) != 8:
        return None
    length = struct.unpack(">Q", header)[0]
    payload = wire_in.read(length)
    if len(payload) != length:
        return None
    return np.load(BytesIO(payload), allow_pickle=False)

def reply(tag, action=None):
    values = np.zeros(7) if action is None else np.asarray(action, dtype=np.float64).reshape(-1)
    if values.shape != (7,) or not np.isfinite(values).all():
        tag, values = 1, np.zeros(7)
    wire_out.write(bytes([tag]) + values.astype(">f8").tobytes())
    wire_out.flush()

path = os.path.join(sys.argv[1], "controller.py")
module_spec = importlib.util.spec_from_file_location("submission_controller", path)
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
controller = module.make_controller()
reply(0)
while True:
    message = read_message()
    if message is None:
        break
    kind = int(message["_kind"])
    if kind == 2:
        break
    try:
        if kind == 0:
            controller.reset(json.loads(bytes(message["context"]).decode()), int(message["seed"]))
            reply(0)
        else:
            observation = {}
            for key in message.files:
                if key == "_kind":
                    continue
                value = message[key]
                observation[key] = float(value) if value.ndim == 0 else value
            observation["target"] = json.loads(bytes(observation.pop("target_json")).decode())
            reply(0, controller.act(observation))
    except Exception:
        import traceback
        traceback.print_exc()
        reply(1)
'''


def _pack(kind: int, values: dict) -> bytes:
    stream = BytesIO()
    np.savez(stream, _kind=np.array(kind), **values)
    payload = stream.getvalue()
    return struct.pack(">Q", len(payload)) + payload


class ControllerFault(RuntimeError):
    pass


class ControllerProcess:
    def __init__(self, source: str, scenario: str, seed: int):
        if os.path.isdir("/tests"):
            if os.geteuid() != 0:
                raise ControllerFault("controller isolation requires a root verifier")
            if os.stat("/tests").st_mode & 0o077:
                raise ControllerFault("private verifier directory is not protected")
        source_stat = os.lstat(source)
        if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_size > spec.MAX_SOURCE_BYTES:
            raise ControllerFault("controller.py must be a bounded regular file")
        self.work = tempfile.mkdtemp(prefix="task10_controller_")
        os.chmod(self.work, 0o755)
        submission = os.path.join(self.work, "submission")
        os.mkdir(submission, 0o755)
        shutil.copyfile(source, os.path.join(submission, "controller.py"), follow_symlinks=False)
        os.chmod(os.path.join(submission, "controller.py"), 0o644)
        model_file = os.path.join(self.work, "nominal.mjb")
        model_lib.save_nominal(model_file)
        os.chmod(model_file, 0o644)
        runner = os.path.join(self.work, "runner.py")
        with open(runner, "w") as handle:
            handle.write(_RUNNER)
        os.chmod(runner, 0o644)
        self.stderr_file = open(os.path.join(self.work, "stderr.log"), "wb")
        environment = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": self.work,
            "TMPDIR": self.work,
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        self.process = subprocess.Popen(
            [sys.executable, runner, submission], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr_file, cwd=self.work,
            env=environment,
        )
        try:
            self._receive(spec.STARTUP_SECONDS)
            context = spec.public_context(model_file, scenario)
            self._send(RESET, {"context": np.frombuffer(json.dumps(context).encode(), dtype=np.uint8), "seed": np.array(seed)})
            self._receive(spec.STARTUP_SECONDS)
        except Exception:
            self.close()
            raise

    def _send(self, kind: int, values: dict) -> None:
        try:
            self.process.stdin.write(_pack(kind, values))
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise ControllerFault("controller pipe closed") from error

    def _receive(self, timeout: float) -> np.ndarray:
        deadline = time.monotonic() + timeout
        payload = b""
        while len(payload) < REPLY_BYTES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.process.kill()
                raise ControllerFault("controller timed out")
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(self.process.stdout.fileno(), REPLY_BYTES - len(payload))
            if not chunk:
                break
            payload += chunk
        if len(payload) != REPLY_BYTES or payload[0] != 0:
            raise ControllerFault("controller returned a fault")
        values = np.frombuffer(payload[1:], dtype=">f8").astype(float)
        if not np.isfinite(values).all():
            raise ControllerFault("controller returned non-finite action")
        return values

    def act(self, observation: dict) -> np.ndarray:
        values = dict(observation)
        values["target_json"] = np.frombuffer(json.dumps(values.pop("target")).encode(), dtype=np.uint8)
        self._send(ACT, values)
        return self._receive(spec.ACT_REPLY_SECONDS)

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._send(STOP, {})
                self.process.wait(timeout=1)
            except Exception:
                self.process.kill()
        self.stderr_file.close()
        shutil.rmtree(self.work, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
