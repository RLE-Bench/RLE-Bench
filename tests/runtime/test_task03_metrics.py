"""Legacy breakdowns remain consistent with committed simulator evidence."""
import json
import os

import pytest

if os.environ.get("RLEBENCH_TEST_TASK") != "task03":
    pytest.skip("family-specific metrics", allow_module_level=True)

from harness.metrics import metrics
from harness.task import configuration
from rlebench.runtime import verify
from rlebench.runtime.engine import Engine
from rlebench.runtime.scoring import score
from test_runtime import FakeWorker
from test_scoring import state


def test_tabletop_quality_and_cumulative_steps_survive_resets():
    s = state("tabletop", dict(quality=.2642033332290309), steps=3284, budget=50000)
    s["results"]["0"]["steps"] = 100  # Only the last episode; dev_steps includes resets.
    result = score(s, metrics=metrics)
    assert result["reward"] == result["success_rate"] == result["quality"] == .2642033332290309
    assert result["component_outcome"] == result["quality"]
    assert result["component_efficiency"] == 0
    assert result["task_score"] == pytest.approx(26.42033332290309)
    assert result["interaction_steps"] == 3284
    assert result["interaction_budget"] == 50000


@pytest.mark.parametrize("unclassified,expected", [(0, 1.), (1, .5)])
def test_cube_counters_and_quality_do_not_bypass_reward_cap(unclassified, expected):
    counts = dict(optimal_qtm=7, actual_qtm=7, unclassified_transitions=unclassified, recoveries=2)
    s = state("pocket", dict(success=True, counters=counts), steps=123, budget=50000)
    s["dev_steps"] = 0
    result = score(s, metrics=metrics)
    assert result["reward"] == expected
    assert result["quality"] == result["component_outcome"] == 1
    assert result["task_score"] == 100
    assert result["interaction_steps"] == 123
    assert {key: result[key] for key in counts} == counts


def test_missing_cube_counters_are_not_fabricated():
    result = score(state("pocket", dict(success=True)), metrics=metrics)
    assert result["reward"] == result["task_score"] == result["quality"] == 0
    assert "optimal_qtm" not in result


def test_hidden_com_includes_failed_and_unreached_trials():
    s = state("hidden_com", dict(quadrant="C", answer="C"), steps=240, budget=12000)
    s["config"]["plan"] *= 3
    s["dev_steps"] = 0
    s["results"]["1"] = dict(evidence=dict(quadrant="A", answer=None), steps=17, ended="sim_error")
    result = score(s, metrics=metrics)
    assert result["reward"] == pytest.approx(1/3)
    assert result["correct"] == result["submitted"] == 1
    assert result["control_steps"] == 257
    assert result["trial_1_reward"] == result["trial_1_correct"] == result["trial_1_submitted"] == 1
    assert result["trial_1_control_steps"] == 240
    assert result["trial_2_reward"] == result["trial_2_submitted"] == 0
    assert result["trial_2_control_steps"] == 17
    assert result["trial_3_reward"] == result["trial_3_control_steps"] == 0


def test_verifier_loads_private_family_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("RLEBENCH_TASK", "TowerMaxHeight")
    engine = Engine(tmp_path, configuration(), FakeWorker)
    engine.s.update(phase="finished", dev_steps=21,
                    results={"0": dict(evidence=dict(quality=.6), steps=20)})
    engine.save("test")
    engine.store.close()
    for name in ("collect", "export_media", "export_diagnostics"):
        monkeypatch.setattr(verify, name, lambda *_: None)
    output = tmp_path / "output"
    verify.verify(tmp_path, output, "attempt")
    result = json.loads((output / "reward.json").read_text())
    assert result["reward"] == result["success_rate"] == .6
    assert result["task_score"] == 60
    assert result["interaction_steps"] == 21
