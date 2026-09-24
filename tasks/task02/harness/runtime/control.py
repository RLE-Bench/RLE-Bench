"""Root-only Harbor hook. Repeated step completion is harmless."""
import argparse
import fcntl
import os
from pathlib import Path
import pwd
import socket
import time

from . import protocol as P
from .handoff import freeze, stop_agents
from .store import read_state


def finish_step(root, step):
    deadline = time.monotonic() + 10
    while True:
        try:
            return _finish_step(root, step)
        except (ConnectionError, FileNotFoundError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(.2)


def _finish_step(root, step):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(150)
        sock.connect(str(root / "control.sock"))
        sock.sendall(P.dumps(dict(op="finish_step", step=step)))
        with sock.makefile("rb") as reader:
            reply = P.loads(reader.readline(P.MAX_REPLY+1))
        if not reply.get("ok"):
            raise RuntimeError("step handoff failed")
        return reply


def collect(root, step):
    agent = pwd.getpwnam("agent")
    with (root / "handoff.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_state(root / "ledger.sqlite")
        if step not in state["finished_steps"]:
            stop_agents(agent.pw_uid)
        if state["config"]["mode"] == "task02" and step == "develop":
            try:
                freeze(root, Path("/workspace"))
            except (ValueError, OSError):
                # An invalid deliverable must not prevent evaluation from starting.
                (root / "handoff").mkdir(exist_ok=True, mode=0o700)
                (root / "handoff_error").write_text("handoff contains unsupported files\n")
        return finish_step(root, step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step")
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("control requires root")
    collect(Path("/var/lib/rlebench"), args.step)


if __name__ == "__main__":
    main()
