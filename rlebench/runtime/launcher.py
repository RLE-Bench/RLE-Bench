"""Container init child: restart the broker, never the container, on failure."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    if os.geteuid() != 0:
        raise SystemExit("launcher requires root")
    os.umask(0o077)
    root = Path("/var/lib/rlebench")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    stopping = False
    children = {}

    def stop(*_):
        nonlocal stopping
        stopping = True
        for child in children.values():
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    modules = {"service": "rlebench.runtime.server"}
    if os.environ.get("RLEBENCH_FAMILY") == "task01" and os.environ.get("RLEBENCH_IMAGE_LEVEL") in ("L2", "L3"):
        modules["perception"] = "harness.perception_service"
    while not stopping:
        for name, module in modules.items():
            if name not in children or children[name].poll() is not None:
                if name == "perception":
                    Path("/run/rlebench/perception.sock").unlink(missing_ok=True)
                with (root / (name+".log")).open("ab", buffering=0) as log:
                    children[name] = subprocess.Popen([sys.executable, "-m", module],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log, cwd="/")
        time.sleep(1)
    for child in children.values():
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


if __name__ == "__main__":
    main()
