"""Stage task10 public assets, isolated verifier harness and Oracle payload."""

from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "tasks/task10"
HARNESS = TASK / "harness"
ROBOT = REPO / "assets/robots/franka_emika_panda"
AGENT_FILES = ("__init__.py", "spec.py", "model.py", "scenarios.py", "practice.py")
VERIFIER_FILES = ("__init__.py", "spec.py", "config.py", "faults.py", "model.py", "scenarios.py", "metrics.py", "controller_process.py", "evaluator.py", "golden.py", "baselines.py", "scorer.py")
FORBIDDEN = ("EVAL_SEEDS", "sample_fault", "GoldenController", "score_submission", "GATE_CAP", "8053", "17749", "29009", "44101", "60647", "77587", "91873", "104729")


def copy_robot(root: Path) -> None:
    target = root / "robots/franka_emika_panda"
    shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROBOT, target)


def stage_package(files: tuple[str, ...], target: Path) -> None:
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for name in files:
        shutil.copy2(HARNESS / name, target / name)


def scan_public(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix in {".stl", ".obj", ".mjb"}:
            continue
        text = path.read_text(errors="ignore")
        for token in FORBIDDEN:
            if token in text:
                raise SystemExit(f"private token {token!r} leaked into {path}")


def main() -> None:
    environment_assets = TASK / "environment/assets"
    stage_package(AGENT_FILES, environment_assets / "harness")
    copy_robot(environment_assets)
    (TASK / "environment/dev_runner.py").write_text("from harness.practice import main\n\nif __name__ == '__main__':\n    main()\n")
    scan_public(TASK / "environment")

    stage_package(VERIFIER_FILES, TASK / "tests/harness")
    copy_robot(TASK / "tests/assets")

    payload = TASK / "solution/payload"
    shutil.rmtree(payload, ignore_errors=True)
    payload.mkdir(parents=True)
    shutil.copy2(HARNESS / "golden.py", payload / "controller.py")
    print("task10 assets staged; public boundary scan passed")


if __name__ == "__main__":
    main()
