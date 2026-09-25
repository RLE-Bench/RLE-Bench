import pytest

from rlebench.runtime.scoring import score


def state(mode, evidence, steps=0, budget=100):
    return dict(config=dict(mode=mode, plan=[{}], budget=budget),
                results={"0": dict(evidence=evidence, steps=steps)}, dev_steps=steps, failures=0)


@pytest.mark.parametrize("steps,expected", [(0, 1), (50, .9), (100, .8)])
def test_speedrun_efficiency(steps, expected):
    assert score(state("task01", dict(success=True), steps))["reward"] == expected
    assert score(state("task01", dict(success=False), steps))["reward"] == 0


@pytest.mark.parametrize("mode,evidence", [
    ("task02", dict(score=.6)), ("tabletop", dict(quality=.6))])
def test_quality_has_no_step_discount(mode, evidence):
    assert score(state(mode, evidence, 99))["reward"] == .6


def test_unreached_trials_still_count_in_denominator():
    s = state("task02", dict(score=.6))
    s["config"]["plan"] *= 3
    assert score(s)["reward"] == pytest.approx(.2)


@pytest.mark.parametrize("actual,unclassified,expected", [(7,0,1), (14,0,.5), (21,0,.25), (0,1,.5), (21,1,.25)])
def test_cube_qtm_quality(actual, unclassified, expected):
    evidence = dict(success=True, counters=dict(optimal_qtm=7, actual_qtm=actual,
                    recoveries=0, unclassified_transitions=unclassified))
    assert score(state("pocket", evidence, 99))["reward"] == expected


def test_missing_cube_counters_cannot_earn_credit():
    assert score(state("pocket", dict(success=True)))["reward"] == 0


@pytest.mark.parametrize("answer,expected", [("A",1), ("B",0), (None,0)])
def test_hidden_mass_uses_private_quadrant(answer, expected):
    result = score(state("hidden_com", dict(quadrant="A",answer=answer)))
    assert result["reward"] == result["success_rate"] == expected
