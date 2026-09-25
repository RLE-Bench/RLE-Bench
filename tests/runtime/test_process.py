import asyncio
import json
import os
from pathlib import Path
import signal
import select
import subprocess
import sys
import time

import pytest

from rlebench.runtime.process import Worker, WorkerFailure


@pytest.fixture
def worker_env(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(map(str, [root, root/'tests/runtime/support'])))


@pytest.mark.parametrize("failure", ["crash", "hang"])
def test_native_worker_failure_is_bounded_and_replaceable(tmp_path, worker_env, failure):
    async def scenario():
        w = Worker(tmp_path, timeout=.15)
        await w.start("fake_adapter", dict(failure=failure))
        with pytest.raises(WorkerFailure) as caught:
            await w.call("act", action=[0])
        assert caught.value.details["kind"] == ("timeout" if failure == "hang" else "exit")
        details = json.loads((tmp_path / "failures.jsonl").read_text())
        assert details["operation"] == "act" and details["timeout_seconds"] == .15
        if failure == "hang":
            assert "fake_adapter.py" in (tmp_path / "worker.log").read_text()
        proc = w.process
        await w.stop()
        assert proc.returncode is not None
        new = Worker(tmp_path)
        await new.start("fake_adapter", {})
        assert (await new.call("act", action=[0]))["score"] == .1
        await new.stop()
    started = time.monotonic()
    asyncio.run(scenario())
    assert time.monotonic()-started < 10


def test_large_reply_survives_interrupted_pipe_write(worker_env):
    proc = subprocess.Popen([sys.executable, "-m", "rlebench.runtime.worker"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(b'{"op":"create","module":"fake_adapter","failure":"large_reply"}\n')
        proc.stdin.flush()
        assert json.loads(proc.stdout.readline())["ok"]
        proc.stdin.write(b'{"op":"observe","spec":{}}\n')
        proc.stdin.flush()
        assert proc.stderr.readline() == b"reply-ready\n"
        # Fill the pipe without reading it, then interrupt its blocking write.
        assert select.select([proc.stdout], [], [], 5)[0]
        time.sleep(.1)
        proc.send_signal(signal.SIGUSR2)
        assert proc.stderr.readline() == b"interrupted\n"
        output, _ = proc.communicate(b'{"op":"reset"}\n{"op":"close"}\n', timeout=5)
        assert proc.returncode == 0
        reply, following = map(json.loads, output.splitlines())
        assert reply["result"]["obs"]["blob"] == "x" * (8 * 1024 * 1024)
        assert following["ok"] and following["result"]["info"]["instruction"] == "fake"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_worker_exits_on_broken_reply_pipe(worker_env):
    proc = subprocess.Popen([sys.executable, "-m", "rlebench.runtime.worker"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(b'{"op":"create","module":"fake_adapter"}\n')
        proc.stdin.flush()
        assert json.loads(proc.stdout.readline())["ok"]
        proc.stdout.close()
        proc.stdin.write(b'{"op":"observe","spec":{}}\n')
        proc.stdin.flush()
        assert proc.wait(timeout=5) != 0
        assert b"BrokenPipeError" in proc.stderr.read()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdin.close()
        proc.stderr.close()


def test_idle_worker_killed_by_signal(tmp_path, worker_env):
    async def scenario():
        w = Worker(tmp_path)
        await w.start("fake_adapter", {})
        os.kill(w.process.pid, signal.SIGKILL)
        with pytest.raises(WorkerFailure):
            await w.call("observe", spec={})
        await w.stop()
    asyncio.run(scenario())
