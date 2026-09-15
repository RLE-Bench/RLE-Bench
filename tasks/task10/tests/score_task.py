"""Harbor verifier entry point for task10."""

from __future__ import annotations

import json
import os
import shutil
import stat
import traceback

ARTIFACTS = "/logs/artifacts"
VERIFIER = "/logs/verifier"
STAGING = "/tmp/task10-submission"


def write(payload: dict) -> None:
    os.makedirs(VERIFIER, exist_ok=True)
    with open(os.path.join(VERIFIER, "reward.json"), "w") as handle:
        json.dump(payload, handle, indent=2)


def stage_controller() -> str:
    source = os.path.join(ARTIFACTS, "controller.py")
    info = os.lstat(source)
    from harness import spec
    if not stat.S_ISREG(info.st_mode) or info.st_size > spec.MAX_SOURCE_BYTES:
        raise ValueError("invalid controller.py")
    shutil.rmtree(STAGING, ignore_errors=True)
    os.mkdir(STAGING, 0o700)
    target = os.path.join(STAGING, "controller.py")
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as reader, open(target, "wb") as writer:
        opened = os.fstat(reader.fileno())
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > spec.MAX_SOURCE_BYTES:
            raise ValueError("invalid controller.py")
        content = reader.read(spec.MAX_SOURCE_BYTES + 1)
        if len(content) > spec.MAX_SOURCE_BYTES:
            raise ValueError("controller.py exceeds size limit")
        writer.write(content)
    return target


def main() -> None:
    try:
        from harness.scorer import score_submission
        report = score_submission(stage_controller())
        reward = {"reward": report["reward"], "load_failures": report["load_failures"]}
        for name, value in report["bands"].items():
            reward[f"band_{name}"] = value
        write(reward)
        with open(os.path.join(VERIFIER, "report.json"), "w") as handle:
            json.dump(report, handle, indent=2)
    except Exception:
        traceback.print_exc()
        write({"reward": 0.0, "scoring_error": 1})


if __name__ == "__main__":
    main()
