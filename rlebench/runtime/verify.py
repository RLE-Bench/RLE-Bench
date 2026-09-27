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
from .handoff import restore, stop_agents
from .scoring import score
from .store import read_state


def export_files(output, directory, paths, index=None):
    agent = pwd.getpwnam("agent")
    owner = output.stat()
    uid, gid = (owner.st_uid, owner.st_gid) if owner.st_uid else (agent.pw_uid, agent.pw_gid)
    # Extract as the host owner; never follow agent-controlled output paths as root.
    script = ('import os,sys,tarfile; os.umask(0o022); os.makedirs(sys.argv[1],exist_ok=True); '
              'tarfile.open(fileobj=sys.stdin.buffer,mode="r|").extractall(sys.argv[1],filter="data")')
    proc = subprocess.Popen(["/usr/local/bin/python", "-I", "-c", script, str(output / directory)],
        stdin=subprocess.PIPE, user=uid, group=gid, extra_groups=[])
    try:
        with tarfile.open(fileobj=proc.stdin, mode="w|") as archive:
            for path in paths:
                archive.add(path, arcname=path.name)
            if index is not None:
                data = json.dumps(index).encode()
                info = tarfile.TarInfo("index.json")
                info.size, info.mode = len(data), 0o644
                archive.addfile(info, io.BytesIO(data))
        proc.stdin.close()
        if proc.wait(timeout=10):
            raise RuntimeError(f"{directory} export failed")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def export_media(root, output, state):
    from .media import evaluation_enabled
    if state["config"]["mode"] not in ("task01", "task02"):
        paths = [p for p in sorted((root / "media").glob("*.mp4"))
                 if not p.is_symlink() and p.is_file() and p.stat().st_size]
        index = dict(enabled=os.environ.get("RLEBENCH_MEDIA", "1") != "0",
                     files=[p.name for p in paths], skipped=[])
    else:
        paths = []
        index = dict(enabled=evaluation_enabled(), files=[], skipped=[])
        if index["enabled"]:
            for trial in sorted(map(int, state["results"])):
                path = root / "media" / f"trial-{trial+1:02d}.mp4"
                metadata = path.with_suffix(".json")
                reason = "missing or interrupted recording"
                try:
                    if metadata.is_symlink():
                        raise ValueError("invalid metadata")
                    record = json.loads(metadata.read_text())
                    if record.get("status") == "complete" and record.get("frames", 0) > 0:
                        if not path.is_symlink() and path.is_file() and path.stat().st_size:
                            paths.append(path)
                            index["files"].append(path.name)
                            continue
                        reason = "finalized video missing or invalid"
                    else:
                        reason = record.get("reason") or "interrupted recording"
                except (OSError, ValueError, TypeError, AttributeError):
                    pass
                index["skipped"].append(dict(name=path.name, reason=reason))
    export_files(output, "media", paths, index)


def export_diagnostics(root, output):
    stop_agents(pwd.getpwnam("agent").pw_uid)
    paths = [root / name for name in ("worker.log", "service.log", "verifier.log", "failures.jsonl")]
    export_files(output, "diagnostics", [p for p in paths if p.is_file() and not p.is_symlink()])


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
                         infrastructure_failures=state["failures"], failures=state.get("failure_details", []))
        try:
            export_media(root, output, state)
        except Exception:
            traceback.print_exc()
            diagnosis["media_ok"] = False
        if state["phase"] == "finished" and diagnosis["handoff_ok"]:
            try:
                export_diagnostics(root, output)
            except Exception:
                traceback.print_exc()
                diagnosis["diagnostics_ok"] = False
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
