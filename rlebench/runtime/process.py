"""Bounded subprocess operations; the parent never loads the simulator."""
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import time

from . import protocol as P

INITIALIZE_SECONDS = 300
OPERATION_SECONDS = 30
FINISH_SECONDS = 120
STOP_SECONDS = 3


class WorkerFailure(RuntimeError):
    def __init__(self, kind="unavailable", **details):
        super().__init__(kind)
        self.details = dict(kind=kind, **details)


class Worker:
    def __init__(self, root, *, timeout=OPERATION_SECONDS):
        self.root = Path(root)
        self.timeout = timeout
        self.process = None

    async def start(self, module, descriptor):
        with (self.root / "worker.log").open("ab", buffering=0) as log:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "rlebench.runtime.worker",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=log, start_new_session=True, limit=P.MAX_REPLY + 1,
                cwd="/", env={**os.environ, "PYTHONSAFEPATH": "1"})
        return await self.call("create", timeout=INITIALIZE_SECONDS, module=module, **descriptor)

    async def call(self, op, timeout=None, **fields):
        started, limit = time.monotonic(), timeout or self.timeout
        try:
            async with asyncio.timeout(limit):
                self.process.stdin.write(P.dumps(dict(op=op, **fields)))
                await self.process.stdin.drain()
                raw = await self.process.stdout.readline()
                if not raw:
                    raise WorkerFailure("exit")
                if len(raw) > P.MAX_REPLY or not raw.endswith(b"\n"):
                    raise WorkerFailure("protocol")
                reply = P.loads(raw)
                if not reply.get("ok"):
                    raise WorkerFailure("operation")
                return reply["result"]
        except (WorkerFailure, OSError, ValueError, KeyError, asyncio.TimeoutError) as exc:
            kind = ("timeout" if isinstance(exc, asyncio.TimeoutError) else
                    exc.details["kind"] if isinstance(exc, WorkerFailure) else
                    "transport" if isinstance(exc, OSError) else "protocol")
            details = dict(operation=op, elapsed_seconds=time.monotonic()-started,
                           timeout_seconds=limit, returncode=self.process.returncode)
            if kind == "timeout" and self.process.returncode is None:
                try:
                    self.process.send_signal(signal.SIGUSR1)
                    await asyncio.sleep(.05)  # Let faulthandler flush before worker teardown.
                except ProcessLookupError:
                    pass
            try:
                with (self.root / "failures.jsonl").open("a") as log:
                    log.write(json.dumps(dict(kind=kind, **details))+"\n")
            except OSError:
                pass
            raise WorkerFailure(kind, **details) from exc

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
