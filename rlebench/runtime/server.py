"""Root-owned public broker and independent lifecycle socket."""
import argparse
import asyncio
import fcntl
import importlib
import math
import os
from pathlib import Path
import signal
import socket
import struct
import traceback

from . import protocol as P
from .engine import Engine
from .process import WorkerFailure

FIELDS = {
    "status": set(), "task_info": set(), "list_tasks": set(),
    "observe": {"spec"}, "step": {"actions", "spec", "trial"}, "reset": {"task", "seed"},
    "end_development": set(), "finish_trial": set(), "next_trial": set(),
    "move": {"position", "quaternion", "gripper", "steps", "trial"},
    "submit": {"quadrant", "trial"}, "recover_drop": set(),
}
READ_ONLY = {"status", "task_info", "list_tasks", "observe"}


class Server:
    def __init__(self, engine, public_path, control_path):
        self.engine = engine
        self.public_path, self.control_path = Path(public_path), Path(control_path)
        self.owner = None
        self.operation = None
        self.control_lock = asyncio.Lock()
        self.servers = []
        self.connections = set()
        self.monitor = None

    async def reply(self, writer, value):
        data = P.dumps(value)
        if len(data) > P.MAX_REPLY:
            data = P.dumps(dict(ok=False, kind="internal_error", error="response unavailable"))
        writer.write(data)
        # A client that stops reading must not hold the control plane hostage.
        await asyncio.wait_for(writer.drain(), 10)

    async def public(self, reader, writer):
        self.connections.add(writer)
        owned = False
        try:
            if self.owner is not None or self.control_lock.locked():
                await self.reply(writer, dict(ok=False, kind="busy", error="another client holds simulator control"))
                return
            self.owner = writer
            owned = True
            await self.reply(writer, dict(ok=True))
            while not writer.is_closing():
                raw = await reader.readline()  # No idle timeout.
                if not raw:
                    break
                accepted = False
                try:
                    if len(raw) > P.MAX_REQUEST:
                        raise P.RequestError("request too large")
                    request = P.loads(raw)
                    op, rid = request.pop("op", None), request.pop("request_id", None)
                    if op not in FIELDS or set(request) - FIELDS[op]:
                        raise P.RequestError("unknown operation or argument")
                    if not isinstance(rid, str) or not 1 <= len(rid) <= 128:
                        raise P.RequestError("request_id is required")
                    if op not in READ_ONLY and not self.engine.store.begin(rid, op):
                        raise P.RequestError("request already received; inspect status", "duplicate_request")
                    accepted = op not in READ_ONLY
                    self.operation = asyncio.create_task(self.engine.dispatch(op, request))
                    try:
                        result = await self.operation
                    except asyncio.CancelledError:
                        if writer.is_closing():
                            break
                        raise
                    finally:
                        self.operation = None
                    if op not in READ_ONLY:
                        self.engine.store.complete(rid)
                    reply = dict(ok=True, **result)
                except (ValueError, TypeError, KeyError) as exc:
                    if accepted and not self.engine.s["pending"]:
                        self.engine.store.complete(rid)
                    reply = dict(ok=False, kind=getattr(exc, "kind", "bad_request"),
                                 error=str(exc) if isinstance(exc, P.RequestError) else "invalid request")
                except Exception:
                    traceback.print_exc()
                    reply = dict(ok=False, kind="internal_error", error="request failed; inspect status")
                await self.reply(writer, reply)
        except (OSError, ValueError, asyncio.TimeoutError, asyncio.CancelledError):
            pass
        finally:
            if owned and self.owner is writer:
                self.owner = None
            self.connections.discard(writer)
            writer.close()

    async def control(self, reader, writer):
        try:
            peer = writer.get_extra_info("socket").getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            if struct.unpack("3i", peer)[1] != os.geteuid():
                raise P.RequestError("control unavailable")
            raw = await asyncio.wait_for(reader.readline(), 5)
            request = P.loads(raw)
            if set(request) != {"op", "step"} or request["op"] != "finish_step":
                raise P.RequestError("unknown control operation")
            async with self.control_lock:
                if request["step"] in self.engine.s["finished_steps"]:
                    await self.reply(writer, dict(ok=True, **self.engine.status()))
                    return
                if self.owner:
                    self.owner.close()
                if self.operation:
                    task, self.operation = self.operation, None
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                if self.engine.s["pending"] or not self.engine.s["ready"]:
                    await self.engine.stop()
                    await self.engine.failed()
                result = await self.engine.finish_step(request["step"])
                await self.reply(writer, dict(ok=True, **result))
        except Exception:
            traceback.print_exc()
            try:
                await self.reply(writer, dict(ok=False, error="control operation failed"))
            except OSError:
                pass
        finally:
            writer.close()

    async def start(self):
        for path, handler, mode in ((self.public_path, self.public, 0o666),
                                    (self.control_path, self.control, 0o600)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.unlink(missing_ok=True)
            server = await asyncio.start_unix_server(handler, path=str(path), limit=P.MAX_REQUEST+1)
            os.chmod(path, mode)
            self.servers.append(server)
        await self.engine.recover()
        self.monitor = asyncio.create_task(self.watch_worker())

    async def watch_worker(self):
        while True:
            await asyncio.sleep(.2)
            worker = self.engine.worker
            process = getattr(worker, "process", None)
            if (process is not None and process.returncode is not None
                    and self.operation is None and not self.control_lock.locked()
                    and self.engine.s["ready"] and not self.engine.s["episode_over"]):
                await self.engine.failed(WorkerFailure("exit", operation="idle", returncode=process.returncode))

    async def close(self):
        if self.monitor:
            self.monitor.cancel()
            await asyncio.gather(self.monitor, return_exceptions=True)
        for server in self.servers:
            server.close()
            await server.wait_closed()
        for writer in tuple(self.connections):
            writer.close()
        if self.operation:
            self.operation.cancel()
            await asyncio.gather(self.operation, return_exceptions=True)
        await self.engine.stop()
        self.engine.store.close()
        for path in (self.public_path, self.control_path):
            path.unlink(missing_ok=True)


async def run(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    with (root / "service.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = importlib.import_module(args.config).configuration()
        multiplier = float(os.environ.get("RLEBENCH_TIMEOUT_MULT", "1"))
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("invalid timeout multiplier")
        config["seconds"] = {key: value*multiplier for key, value in config.get("seconds", {}).items()}
        engine = Engine(root, config)
        server = Server(engine, args.socket, root / "control.sock")
        await server.start()
        done = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, done.set)
        try:
            await done.wait()
        finally:
            await server.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="harness.task")
    parser.add_argument("--root", default="/var/lib/rlebench")
    parser.add_argument("--socket", default="/run/rlebench/sim.sock")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
