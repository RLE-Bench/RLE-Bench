"""Task 07 verifier scoring entry point.

This is copied to ``/tests/score_task.py`` in the separate verifier image.
It stages untrusted agent artifacts, reruns the policy on hidden evaluation
seeds, and writes ``reward.json`` and ``report.json``.
"""
from __future__ import annotations

import json
import os
import shutil
import traceback

ARTIFACTS = os.environ.get("RLEBENCH_ARTIFACTS", "/logs/artifacts")
STAGING = os.environ.get("RLEBENCH_STAGING", "/tmp/submission")
VERIFIER_DIR = os.environ.get("RLEBENCH_VERIFIER_DIR", "/logs/verifier")


def _write(reward, report=None):
    os.makedirs(VERIFIER_DIR, exist_ok=True)
    with open(os.path.join(VERIFIER_DIR, "reward.json"), "w") as stream:
        json.dump(reward, stream, indent=2)
    if report is not None:
        with open(os.path.join(VERIFIER_DIR, "report.json"), "w") as stream:
            json.dump(report, stream, indent=2, default=str)


def _media():
    """Episode videos for a human reader (RLEBENCH_MEDIA=0 turns them off);
    never an input to the score."""
    try:
        from rlebench.core.media import Media
        return Media.verifier(VERIFIER_DIR)
    except Exception as e:  # noqa: BLE001
        print(f"media disabled: {e}")
        return None


def main():
    try:
        if os.geteuid() == 0:
            for directory in ("/tests", VERIFIER_DIR):
                if os.path.exists(directory) and os.stat(directory).st_mode & 0o077:
                    raise RuntimeError(f"private verifier directory is not protected: {directory}")
        if os.path.exists(STAGING):
            shutil.rmtree(STAGING)
        source = os.path.join(ARTIFACTS, "policy")
        if os.path.isdir(source):
            shutil.copytree(source, STAGING, symlinks=True)
        else:
            os.makedirs(STAGING)
        os.chmod(STAGING, 0o700)

        from harness.scorer import score_submission
        media = _media()
        report = score_submission(STAGING, media=media)

        reward = {"reward": float(report.get("reward", 0.0)),
                  "raw_total": float(report.get("raw_total", 0.0)),
                  "gated": int(report.get("gated", True))}
        for key in ("clear_frac", "parts_cleared", "n_parts", "speed_merit",
                    "throughput_ppm", "makespan_s", "floor_drops",
                    "damage_events", "bin_hits", "policy_faults",
                    "episode_wall_s_total", "policy_wall_s_total",
                    "harness_wall_s_total"):
            if report.get(key) is not None:
                reward[key] = float(report[key])
        for name, ok in report["gate"].items():
            reward[f"gate_{name}"] = int(bool(ok))
        _write(reward, report)
        print(json.dumps(report, indent=2, default=str))
        if media is not None:
            media.report()
    except Exception:
        traceback.print_exc()
        _write({"reward": 0.0, "scoring_error": 1})


if __name__ == "__main__":
    main()
