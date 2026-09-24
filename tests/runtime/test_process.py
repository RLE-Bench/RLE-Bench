import asyncio
import os
from pathlib import Path
import signal
import time

import pytest

from harness.runtime.process import Worker, WorkerFailure


@pytest.fixture
def worker_env(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    family = os.environ.get("RLEBENCH_TEST_TASK", "task01")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(map(str, [root/'tasks'/family, root/'tests/runtime/support'])))


@pytest.mark.parametrize("failure", ["crash", "hang"])
def test_native_worker_failure_is_bounded_and_replaceable(tmp_path, worker_env, failure):
    async def scenario():
        w = Worker(tmp_path, timeout=.15)
        await w.start("fake_adapter", dict(failure=failure))
        with pytest.raises(WorkerFailure):
            await w.call("act", action=[0])
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


def test_idle_worker_killed_by_signal(tmp_path, worker_env):
    async def scenario():
        w = Worker(tmp_path)
        await w.start("fake_adapter", {})
        os.kill(w.process.pid, signal.SIGKILL)
        with pytest.raises(WorkerFailure):
            await w.call("observe", spec={})
        await w.stop()
    asyncio.run(scenario())
