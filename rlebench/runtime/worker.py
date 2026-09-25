"""Single-threaded simulator process. Its stdout is reserved for private IPC."""
import ctypes
import faulthandler
import importlib
import os
import signal
import sys
import traceback

from . import protocol as P


def main():
    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # Die with the supervisor, including native hangs.
    if parent == 1 or os.getppid() != parent:
        return
    output = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # Simulator imports can print; none of that is protocol data.
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    adapter = None
    def shutdown(*_):
        # Python-level shutdown closes encoders; a native hang is killed by the parent.
        if adapter is not None:
            adapter.close()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, shutdown)
    for line in sys.stdin.buffer:
        try:
            message = P.loads(line)
            op = message.pop("op")
            if op == "create":
                module = importlib.import_module(message.pop("module"))
                adapter = module.Adapter(**message)
                result = adapter.reset()
            elif op == "close":
                if adapter is not None:
                    adapter.close()
                break
            else:
                result = getattr(adapter, op)(**message)
            reply = dict(ok=True, result=result)
        except Exception:
            traceback.print_exc()
            reply = dict(ok=False)
        data = P.dumps(reply)
        if len(data) > P.MAX_REPLY:
            data = P.dumps(dict(ok=False))
        output.write(data)
        output.flush()


if __name__ == "__main__":
    main()
