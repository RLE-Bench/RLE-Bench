"""Exercise the task10 controller boundary as root, like the verifier image."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from harness import spec
from harness.controller_process import ControllerFault, ControllerProcess


def main() -> None:
    if os.geteuid() == 0 and Path("/tests").is_dir():
        os.chmod("/tests", 0o700)
    root = Path(tempfile.mkdtemp(prefix="task10_isolation_"))
    private = root / "private"
    private.mkdir(mode=0o700)
    secret = private / "secret"
    secret.write_text("verifier-only")
    policy = root / "controller.py"
    expected_uid = 65534 if os.geteuid() == 0 else os.geteuid()
    protected_paths = [str(secret), f"/proc/{os.getpid()}/mem"] if os.geteuid() == 0 else []
    policy.write_text(f'''import os
import mujoco
class Controller:
 def reset(self, context, seed):
  assert os.geteuid() == {expected_uid}
  assert "PYTHONPATH" not in os.environ
  for path in {protected_paths!r}:
   try:
    open(path, "rb").read(1)
   except (PermissionError, OSError):
    pass
   else:
    raise AssertionError("private file readable: " + path)
  try:
   import harness.config
  except (ModuleNotFoundError, PermissionError):
   pass
  else:
   raise AssertionError("verifier harness importable")
  try:
   os.fork()
  except OSError:
   pass
  else:
   raise AssertionError("fork succeeded")
  local = mujoco.MjModel.from_binary_path(context["model_file"])
  local.body_mass[:] = 999
 def act(self, observation): return [0.25] * 7
def make_controller(): return Controller()
''')
    try:
        with ControllerProcess(str(policy), "free_space", 1) as process:
            action = process.act({"target": {"position": [0, 0, 0], "normal_force": 0, "phase": "evaluation"}})
            assert np.allclose(action, 0.25)
        assert secret.read_text() == "verifier-only"
        original_timeout = spec.ACT_REPLY_SECONDS
        spec.ACT_REPLY_SECONDS = 0.15
        for body in ("return [float('nan')] * 7", "return [1, 2]", "raise RuntimeError('bad')", "\n  while True: pass"):
            policy.write_text(f'''class Controller:
 def reset(self, context, seed): pass
 def act(self, observation):
  {body}
def make_controller(): return Controller()
''')
            try:
                with ControllerProcess(str(policy), "free_space", 1) as process:
                    process.act({"target": {"position": [0, 0, 0], "normal_force": 0, "phase": "evaluation"}})
            except ControllerFault:
                pass
            else:
                raise AssertionError(f"invalid controller accepted: {body}")
        spec.ACT_REPLY_SECONDS = original_timeout
        print("PASS: controller boundary rejects private reads/imports, fork, local-model mutation, malformed actions and hangs")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
