"""Private, immutable development artifacts for independent transfer trials."""
import os
from pathlib import Path
import pwd
import shutil
import signal
import stat
import time


def stop_agents(uid):
    if uid == 0:
        raise ValueError("agent must not be root")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for path in Path("/proc").glob("[0-9]*/status"):
            try:
                fields = dict(line.split(":", 1) for line in path.read_text().splitlines())
                if int(fields["Uid"].split()[0]) == uid:
                    os.kill(int(path.parent.name), sig)
            except (FileNotFoundError, ProcessLookupError):
                pass
        time.sleep(.1)


def copy_regular(source, target):
    """Do not follow links from an agent-controlled tree, including the tree root."""
    metadata = source.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("handoff must be a directory")
    target.mkdir(mode=0o700)
    for item in source.iterdir():
        mode = item.lstat().st_mode
        destination = target / item.name
        if stat.S_ISDIR(mode):
            copy_regular(item, destination)
        elif stat.S_ISREG(mode):
            # Agents have already been stopped; O_NOFOLLOW also rejects link substitutions.
            fd = os.open(item, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream, destination.open("xb") as out:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("handoff contains a special file")
                shutil.copyfileobj(stream, out)
            destination.chmod(0o700 if mode & 0o111 else 0o600)
        else:
            raise ValueError("handoff contains links or special files")


def freeze(root, workspace):
    snapshot = root / "handoff"
    if snapshot.exists():
        return
    source = workspace / "agent_harness"
    temporary = root / "handoff.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    if source.exists():
        copy_regular(source, temporary)
    else:
        temporary.mkdir(mode=0o700)
    temporary.rename(snapshot)


def restore(root, workspace, uid, gid):
    for item in workspace.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()
    target = workspace / "agent_harness"
    copy_regular(root / "handoff", target)
    for item in [target, *target.rglob("*")]:
        os.chown(item, uid, gid, follow_symlinks=False)
    # Keep CLI installations, remove their conversation stores between fresh agents.
    home = Path(pwd.getpwuid(uid).pw_dir)
    for relative in (".claude/projects", ".codex/sessions", ".codex/history.jsonl",
                     ".gemini/tmp", ".grok/sessions"):
        path = home / relative
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
