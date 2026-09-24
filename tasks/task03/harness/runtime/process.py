"""Bounded subprocess operations; the parent never loads the simulator."""
import asyncio
import os
from pathlib import Path
import signal
import sys

from . import protocol as P

INITIALIZE_SECONDS = 300
OPERATION_SECONDS = 30
FINISH_SECONDS = 120
STOP_SECONDS = 3


class WorkerFailure(RuntimeError):
    pass


class Worker:
    def __init__(self, root, *, timeout=OPERATION_SECONDS):
        self.root = Path(root)
        self.timeout = timeout
        self.process = None

    async def start(self, module, descriptor):
        with (self.root / "worker.log").open("ab", buffering=0) as log:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "harness.runtime.worker",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=log, start_new_session=True, limit=P.MAX_REPLY + 1,
                cwd="/", env={**os.environ, "PYTHONSAFEPATH": "1"})
        return await self.call("create", timeout=INITIALIZE_SECONDS, module=module, **descriptor)

    async def call(self, op, timeout=None, **fields):
        try:
            async with asyncio.timeout(timeout or self.timeout):
                self.process.stdin.write(P.dumps(dict(op=op, **fields)))
                await self.process.stdin.drain()
                raw = await self.process.stdout.readline()
                if not raw or len(raw) > P.MAX_REPLY:
                    raise WorkerFailure("worker exited")
                reply = P.loads(raw)
                if not reply.get("ok"):
                    raise WorkerFailure("worker operation failed")
                return reply["result"]
        except (OSError, ValueError, asyncio.TimeoutError) as exc:
            raise WorkerFailure("worker unavailable") from exc

    async def stop(self):
        proc, self.process = self.process, None
        if proc is None:
            return
        if proc.returncode is None:
            try:
                proc.stdin.write(P.dumps(dict(op="close")))
                proc.stdin.close()
                await asyncio.wait_for(proc.wait(), STOP_SECONDS)
            except (OSError, asyncio.TimeoutError):
                pass
        # Reap descendants as well, including media encoders.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()
