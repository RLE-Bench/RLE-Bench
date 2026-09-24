"""Fault injection runs in the actual worker subprocess."""
import os
import time


class Adapter:
    def __init__(self, **descriptor):
        self.failure = descriptor.get("failure")
        self.steps = 0
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
        return dict(obs={"position": self.steps})

    def close(self):
        pass
