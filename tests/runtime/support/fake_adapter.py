"""Fault injection runs in the actual worker subprocess."""
import os
import signal
import sys
import time


class Adapter:
    def __init__(self, **descriptor):
        self.failure = descriptor.get("failure")
        self.steps = 0
        if self.failure == "large_reply":
            signal.signal(signal.SIGUSR2, lambda *_: print("interrupted", file=sys.stderr, flush=True))
        if self.failure == "create":
            os._exit(9)

    def reset(self, seed=None):
        return dict(info=dict(instruction="fake"), evidence=dict(success=False))

    def act(self, **kw):
        if self.failure == "hang":
            time.sleep(60)
        if self.failure == "crash":
            os._exit(9)
        self.steps += 1
        return dict(success=False, score=self.steps/10)

    def observe(self, spec):
        if self.failure == "large_reply":
            print("reply-ready", file=sys.stderr, flush=True)
            return dict(obs={"blob": "x" * (8 * 1024 * 1024)})
        return dict(obs={"position": self.steps})

    def close(self):
        pass
