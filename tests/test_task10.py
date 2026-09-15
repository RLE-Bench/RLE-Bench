"""Host checks for task10's deterministic verifier and public boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TASK = REPO / "tasks/task10"
sys.path.insert(0, str(TASK))


def test_fault_generation_is_deterministic():
    from harness.faults import sample_fault
    assert sample_fault(991) == sample_fault(991)
    assert sample_fault(991) != sample_fault(992)


def test_metric_endpoints_are_analytic():
    from harness import config, metrics
    assert metrics.tracking_score(config.TRACK_EXCELLENT_M) == 1.0
    assert metrics.tracking_score(config.MAX_POSITION_RMSE_M) == 0.0
    assert metrics.contact_score(0.022, config.CONTACT_FORCE_EXCELLENT_N, 18.0) == 1.0
    assert metrics.recovery_score(config.TRACK_EXCELLENT_M, config.RECOVERY_EXCELLENT_S) == 1.0


def test_public_tree_contains_no_private_tokens():
    subprocess.run([sys.executable, str(TASK / "build_assets.py")], cwd=REPO, check=True)
    public = TASK / "environment"
    forbidden = ("EVAL_SEEDS", "sample_fault", "GoldenController", "score_submission", "GATE_CAP")
    text = "\n".join(path.read_text(errors="ignore") for path in public.rglob("*") if path.is_file() and path.suffix in {".py", ".md", ".toml"})
    assert not any(token in text for token in forbidden)


def test_model_and_scenario_replay_are_identical():
    from harness import model, scenarios, spec
    first = model.build_model(1.7)
    second = model.build_model(1.7)
    assert first.nq == second.nq == 7
    first_payload = json.dumps([scenarios.target("recovery", step)["position"].tolist() for step in range(spec.EVALUATION_STEPS)], sort_keys=True).encode()
    second_payload = json.dumps([scenarios.target("recovery", step)["position"].tolist() for step in range(spec.EVALUATION_STEPS)], sort_keys=True).encode()
    assert first.body_mass.tolist() == second.body_mass.tolist()
    assert hashlib.sha256(first_payload).digest() == hashlib.sha256(second_payload).digest()


def test_golden_clears_reference_bar():
    from harness import config
    from harness.evaluator import evaluate_factory
    from harness.golden import make_controller
    report = evaluate_factory(make_controller, seeds=config.EVAL_SEEDS[:2])
    assert report["reward"] >= 0.72
    assert not any(episode["unsafe"] for episode in report["episodes"])


def test_nominal_stays_below_golden():
    from harness.baselines import NominalController
    from harness.evaluator import evaluate_factory
    from harness.golden import make_controller
    seeds = (8053, 60647)
    nominal = evaluate_factory(NominalController, seeds=seeds)["reward"]
    golden = evaluate_factory(make_controller, seeds=seeds)["reward"]
    assert nominal <= 0.30
    assert golden - nominal >= 0.45
