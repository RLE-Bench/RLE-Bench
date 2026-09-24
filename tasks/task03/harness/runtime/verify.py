"""Verifier-only reward writer, independent of simulator availability."""
import argparse
import io
import json
import os
from pathlib import Path
import pwd
import subprocess
import tarfile
import tempfile
import traceback

from .control import collect
from .handoff import restore
from .scoring import score
from .store import read_state


def export_media(root, output):
    agent = pwd.getpwnam("agent")
    owner = output.stat()
    uid, gid = (owner.st_uid, owner.st_gid) if owner.st_uid else (agent.pw_uid, agent.pw_gid)
    # Harbor archives bind-mounted directories before its final ownership cleanup.
    # Extract as their non-root host owner; never give agent paths root write access.
    script = ('import os,sys,tarfile; os.umask(0o022); os.makedirs(sys.argv[1],exist_ok=True); '
              'tarfile.open(fileobj=sys.stdin.buffer,mode="r|").extractall(sys.argv[1],filter="data")')
    proc = subprocess.Popen(["/usr/local/bin/python", "-I", "-c", script, str(output / "media")],
        stdin=subprocess.PIPE, user=uid, group=gid, extra_groups=[])
    try:
        with tarfile.open(fileobj=proc.stdin, mode="w|") as archive:
            files = []
            for path in sorted((root / "media").glob("*.mp4")):
                if not path.is_symlink() and path.stat().st_size:
                    archive.add(path, arcname=path.name)
                    files.append(path.name)
            index = json.dumps(dict(enabled=os.environ.get("RLEBENCH_MEDIA", "1") != "0",
                                    files=files, skipped=[])).encode()
            info = tarfile.TarInfo("index.json")
            info.size, info.mode = len(index), 0o644
            archive.addfile(info, io.BytesIO(index))
        proc.stdin.close()
        if proc.wait(timeout=10):
            raise RuntimeError("media export failed")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")
    temporary.chmod(0o644)
    temporary.replace(path)


def verify(root, output, step):
    output.mkdir(parents=True, exist_ok=True)
    diagnosis = dict(ledger_ok=False, handoff_ok=False)
    result = {"reward": 0., "infrastructure_failures": 1}
    try:
        try:
            collect(root, step)
            diagnosis["handoff_ok"] = True
        except Exception:
            traceback.print_exc()
        state = read_state(root / "ledger.sqlite")
        result = score(state, (float(os.environ.get("RLEBENCH_W_OUTCOME", ".8")),
                               float(os.environ.get("RLEBENCH_W_EFFICIENCY", ".2"))))
        diagnosis.update(ledger_ok=True, phase=state["phase"],
                         infrastructure_failures=state["failures"])
        try:
            export_media(root, output)
        except Exception:
            traceback.print_exc()
            diagnosis["media_ok"] = False
        if state["config"]["mode"] == "task02" and state["phase"] != "finished":
            agent = pwd.getpwnam("agent")
            restore(root, Path("/workspace"), agent.pw_uid, agent.pw_gid)
    except Exception:
        traceback.print_exc()
        diagnosis["error"] = "verification or handoff unavailable"
    write_json(output / "diagnosis.json", diagnosis)
    write_json(output / "reward.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step")
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("verifier requires root")
    root = Path("/var/lib/rlebench")
    with (root / "verifier.log").open("a") as log:
        import contextlib
        with contextlib.redirect_stderr(log):
            verify(root, Path("/logs/verifier"), args.step)


if __name__ == "__main__":
    main()
