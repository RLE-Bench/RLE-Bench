"""Task09 verifier entry point (runs inside the verifier container).

Stages the agent's three artifact directories, scores each as untrusted
input in its own interpreter, averages their rewards, and writes:
  /logs/verifier/reward.json   numeric-only summary (Harbor reads this)
  /logs/verifier/report.json   full structured report for humans
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback

VARIANTS = ("franka", "ur5e", "xarm7")
ARTIFACTS = os.environ.get("RLEBENCH_ARTIFACTS", "/logs/artifacts")
STAGING = os.environ.get("RLEBENCH_STAGING", "/tmp/submissions")
VERIFIER_DIR = os.environ.get("RLEBENCH_VERIFIER_DIR", "/logs/verifier")
TEST_ROOT = os.environ.get("RLEBENCH_TEST_ROOT", "/tests")


def _write(reward: dict, report: dict | None = None) -> None:
    os.makedirs(VERIFIER_DIR, exist_ok=True)
    with open(os.path.join(VERIFIER_DIR, "reward.json"), "w") as f:
        json.dump(reward, f, indent=2)
    if report is not None:
        with open(os.path.join(VERIFIER_DIR, "report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)


def _has_links(root: str) -> bool:
    return any(os.path.islink(os.path.join(directory, name))
               for directory, dirs, files in os.walk(root)
               for name in dirs + files)


def _score_variant_process(variant: str, submission: str,
                           output_path: str) -> None:
    """Score one variant in a fresh interpreter bound to its public spec."""
    from harness.codesign_scorer import score_submission

    sref_xml = os.path.join(
        TEST_ROOT, "models", "gello_codesign", variant, "lead_sref.xml")
    report = score_submission(
        submission, sref_xml=sref_xml, variant=variant)
    report["variant"] = variant
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, default=str)


def _failed_report(variant: str, message: str) -> dict:
    return {
        "variant": variant,
        "reward": 0.0,
        "raw_total": 0.0,
        "gated": True,
        "stages": {},
        "checkpoints": {},
        "errors": [message],
    }


def main() -> None:
    try:
        if os.path.exists(STAGING):
            shutil.rmtree(STAGING)
        os.makedirs(STAGING)
        os.chmod(STAGING, 0o700)

        reports = {}
        reward = {}
        for variant in VARIANTS:
            source = os.path.join(ARTIFACTS, variant)
            submission = os.path.join(STAGING, variant)
            if os.path.isdir(source) and not os.path.islink(source):
                shutil.copytree(source, submission, symlinks=True)
                if _has_links(submission):   # root must not follow agent links
                    shutil.rmtree(submission)
            if not os.path.isdir(submission):
                os.makedirs(submission)

            output_path = os.path.join(STAGING, f"{variant}-report.json")
            env = os.environ.copy()
            env["RLEBENCH_GELLO_VARIANT"] = variant
            completed = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--score-variant",
                 variant, submission, output_path],
                env=env, check=False)
            if completed.returncode == 0 and os.path.isfile(output_path):
                with open(output_path, encoding="utf-8") as stream:
                    report = json.load(stream)
            else:
                report = _failed_report(
                    variant,
                    f"variant scorer exited with status {completed.returncode}")
            reports[variant] = report
            reward[f"{variant}_reward"] = float(report["reward"])
            reward[f"{variant}_raw_total"] = float(report["raw_total"])
            reward[f"{variant}_gated"] = int(report["gated"])
            reward[f"{variant}_validity_deduction"] = float(report.get("validity_deduction", 0.0))
            for checkpoint, value in report.get("deductions", {}).items():
                reward[f"{variant}_deduction_{checkpoint}"] = float(value)
            for stage, value in report["stages"].items():
                reward[f"{variant}_stage_{stage}"] = float(value)
            for checkpoint, value in report["checkpoints"].items():
                reward[f"{variant}_cp_{checkpoint}"] = float(value)

        reward["reward"] = sum(
            reward[f"{variant}_reward"] for variant in VARIANTS
        ) / len(VARIANTS)
        combined = {"reward": reward["reward"], "variants": reports}
        _write(reward, combined)
        print(json.dumps(combined, indent=2, default=str))
        # post-reward renders of each submitted device (informational; the
        # reward is already on disk, so a render failure costs nothing)
        try:
            from harness.render import render_submission
            from rlebench.core.media import Media
            media = Media.verifier(VERIFIER_DIR)
            for variant in VARIANTS:
                lead = os.path.join(STAGING, variant, "lead.xml")
                if os.path.exists(lead):
                    media.run(variant, render_submission, lead, variant)
            media.report()
        except Exception as e:
            print(f"render skipped: {e}")
    except Exception:
        traceback.print_exc()
        _write({"reward": 0.0, "scoring_error": 1})


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--score-variant":
        try:
            _score_variant_process(sys.argv[2], sys.argv[3], sys.argv[4])
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
    else:
        main()
