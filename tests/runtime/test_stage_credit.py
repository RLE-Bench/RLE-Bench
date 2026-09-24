"""Tests for stage credit -- the only genuinely new scoring logic in task02.

Two layers, because the two things being tested fail differently:

  * the ARITHMETIC (`score_trial`, `merge_best`) is pure and is tested exhaustively
    here, with no simulator. Its properties -- zero for a do-nothing trial, one for a
    solved one -- are what make the reward mean anything.
  * the MIRROR (each task's stage function against RoboCasa's own `_check_success`) can
    only be tested against a live environment, and is marked `simulator`. It skips on a
    box without RoboCasa assets and runs in the task01 image, where the equivalence is
    the tripwire for a RoboCasa upgrade silently changing what the reward measures.
"""

from __future__ import annotations
import os
import pytest
if os.environ.get("RLEBENCH_TEST_TASK") != "task02":
    pytest.skip("family-specific metrics", allow_module_level=True)


import pytest

from harness import config as C
from harness import stages as S


def vec(*flags: bool) -> S.Stages:
    return [(f"s{i}", bool(f)) for i, f in enumerate(flags)]


# -- the registry agrees with the split --------------------------------------

def test_every_evaluation_task_has_a_stage_function():
    """A missing entry would silently reduce that trial to binary success -- the exact
    thing stage credit exists to avoid, and invisible in the reward.

    Subset, not equality: the registry covers every activity group, and RLEBENCH_GROUP
    selects one of them into EVAL_TASKS."""
    assert set(C.EVAL_TASKS) <= set(S.STAGE_FNS)


def test_every_stage_function_declares_its_names():
    assert set(S.STAGE_FNS) == set(S.STAGE_NAMES)


def test_every_task_is_graded_on_at_least_two_conjuncts():
    """A one-conjunct task is binary however it is scored, and would mean the mirror
    missed something: every composite predicate in the split is a conjunction."""
    for task, names in S.STAGE_NAMES.items():
        assert len(names) >= 2, task
        assert len(set(names)) == len(names), task


# -- the zero floor ----------------------------------------------------------

def test_a_trial_that_changed_nothing_scores_zero():
    """The property the whole reward rests on. `gripper_obj_far` is true at reset for
    nearly every task, so a raw fraction would pay for doing nothing."""
    reset = vec(False, True)
    assert S.score_trial(reset, reset) == 0.0


def test_a_reset_with_one_free_conjunct_scores_zero():
    """A scene that arrives with one conjunct already satisfied must not pay for it."""
    reset = vec(False, False, True)
    assert S.score_trial(reset, reset) == 0.0


def test_a_task_already_satisfied_at_reset_scores_zero_not_one():
    """Unreachable through config.EVAL_TASKS, which is audited, but scoring it as a
    free 1.0 would be the worst failure available here."""
    reset = vec(True, True)
    assert S.score_trial(reset, reset) == 0.0


# -- the ceiling -------------------------------------------------------------

def test_all_conjuncts_satisfied_scores_one():
    assert S.score_trial(vec(False, True), vec(True, True)) == 1.0


def test_success_short_circuits_to_one():
    """The environment's predicate outranks this module. If they ever disagree the
    environment wins and the equivalence test has a bug to report."""
    assert S.score_trial(vec(False, False), vec(False, False), success=True) == 1.0


# -- partial credit ----------------------------------------------------------

def test_half_the_available_conjuncts_scores_a_half():
    assert S.score_trial(vec(False, False), vec(True, False)) == 0.5


def test_credit_is_normalised_against_what_was_available():
    """Two of three conjuncts satisfied, one of which was free at reset, means ONE of
    the two available ones was earned."""
    assert S.score_trial(vec(False, False, True), vec(True, False, True)) == 0.5


def test_regression_below_the_reset_state_is_floored_at_zero():
    """Knocking a satisfied conjunct back out must not produce a negative score."""
    assert S.score_trial(vec(False, True), vec(False, False)) == 0.0


def test_regression_costs_the_credit_it_undoes():
    reset = vec(False, False, True)
    assert S.score_trial(reset, vec(True, True, False)) == 0.5


# -- degenerate input --------------------------------------------------------

def test_an_unreadable_trial_scores_zero():
    assert S.score_trial(None, None) == 0.0
    assert S.score_trial(vec(False), None) == 0.0


def test_a_missing_reset_sample_is_treated_as_nothing_satisfied():
    """Conservative in the agent's favour only where it cannot help it: with no reset
    sample there is no baseline to subtract, so credit is the raw fraction."""
    assert S.score_trial(None, vec(True, False)) == 0.5


# -- best-so-far -------------------------------------------------------------

def test_merge_best_keeps_the_fuller_vector():
    assert S.merge_best(vec(True, False), vec(True, True)) == vec(True, True)


def test_merge_best_does_not_take_a_per_conjunct_union():
    """The one that matters: unioning conjuncts satisfied at DIFFERENT times would
    manufacture a solved trial that never existed."""
    merged = S.merge_best(vec(True, False), vec(False, True))
    assert S.satisfied(merged) == 1


def test_merge_best_tolerates_missing_samples():
    assert S.merge_best(None, vec(True)) == vec(True)
    assert S.merge_best(vec(True), None) == vec(True)
    assert S.merge_best(None, None) is None


def test_progress_that_is_later_undone_still_counts():
    """Success is detected after every step and ends the trial there, so partial credit
    read off the final frame alone would be held to a stricter standard."""
    best = S.merge_best(S.merge_best(vec(False, False), vec(True, False)),
                        vec(False, False))
    assert S.score_trial(vec(False, False), best) == 0.5


# -- reading stages off an environment ---------------------------------------

class _Boom:
    def __getattr__(self, name):
        raise RuntimeError("half-built env")


def test_read_stages_never_raises():
    """A trial that cannot be measured is scored as unmeasured, never a crash in the
    daemon doing the measuring."""
    assert S.read_stages(C.EVAL_TASKS[0], _Boom()) is None
    assert S.read_stages(C.EVAL_TASKS[0], None) is None
    assert S.read_stages("NotATask", object()) is None


def test_has_stages_matches_the_registry():
    assert S.has_stages(C.EVAL_TASKS[0])
    assert not S.has_stages("CloseDrawer")


def test_no_evaluation_task_is_effectively_binary():
    """After reset-normalisation the denominator is the conjuncts NOT already satisfied
    at reset. The previous split had three tasks with only ONE, making them binary
    however they were scored; this asserts the current split has none."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "tasks" / "task02" / "dev" / "data" / "task02_stage_baseline.json"
    if not path.is_file():
        pytest.skip("baseline not measured on this box")
    rows = {row["task"]: row for row in json.load(path.open())}
    # The adopted split's own task must be measured, or the loop below passes on a task
    # nobody has counted the conjuncts of -- which is how FryingPanAdjustment shipped
    # with one. Scoped to EVAL_TASKS, not the whole registry: a mirror that no group
    # currently grades costs nothing to leave unmeasured.
    missing = sorted(set(C.EVAL_TASKS) - set(rows))
    assert not missing, (f"{missing} has no measured reset baseline; re-run "
                         "the stage-baseline sweep")
    for row in rows.values():
        available = row["total"] - row["baseline"]
        if row["task"] in S.LOW_RESOLUTION:
            # Both directions, so the exclusion list cannot rot: a mirror that gains
            # resolution must be taken off it rather than quietly kept.
            assert available < 2, f"{row['task']} is no longer low-resolution"
            continue
        assert available >= 2, f"{row['task']} has only {available} available conjunct"


# -- the disjunctive predicate -----------------------------------------------
#
# AfterwashSorting is graded on a PARTITION, not on named bowls:
#
#     (not water_on) and ((f1 & f2 in bowl1 and f3 in bowl2)
#                      or (f1 & f2 in bowl2 and f3 in bowl1))
#
# A stage function that named a bowl would pass at reset (everything false) and fail only
# when an agent happened to choose the other bowl. These drive the logic directly, with a
# fake, because physics makes the solved state awkward to reach in a test.

class _FakeSort:
    """Minimal stand-in: `placement` maps food -> bowl, or None for 'held'."""

    def __init__(self, placement, water_on):
        self.placement = placement
        self._water_on = water_on
        self.sink = self

    def get_handle_state(self, env=None):
        return {"water_on": self._water_on}


def _stages_for(placement, water_on, monkeypatch):
    fake = _FakeSort(placement, water_on)
    monkeypatch.setattr(
        S, "_ou",
        lambda: type("OU", (), {
            "check_obj_in_receptacle": staticmethod(
                lambda env, food, bowl: env.placement.get(food) == bowl)})())
    return S.afterwash_sorting(fake)


@pytest.mark.parametrize("pair,odd", [("bowl1", "bowl2"), ("bowl2", "bowl1")])
def test_either_partition_counts_as_solved(pair, odd, monkeypatch):
    """Both branches must score identically. This is the assertion a bowl-named
    implementation fails."""
    st = _stages_for({"food1": pair, "food2": pair, "food3": odd}, False, monkeypatch)
    assert all(v for _, v in st), st


def test_all_three_in_one_bowl_is_not_solved(monkeypatch):
    """The pair is together, but nothing is separated -- partial, not solved."""
    st = _stages_for({"food1": "bowl1", "food2": "bowl1", "food3": "bowl1"},
                     False, monkeypatch)
    assert dict(st)["pair_together"] and not dict(st)["odd_one_separated"]


def test_water_left_running_loses_only_that_conjunct(monkeypatch):
    st = _stages_for({"food1": "bowl1", "food2": "bowl1", "food3": "bowl2"},
                     True, monkeypatch)
    assert not dict(st)["water_off"]
    assert dict(st)["pair_together"] and dict(st)["odd_one_separated"]


def test_split_pair_is_not_together(monkeypatch):
    st = _stages_for({"food1": "bowl1", "food2": "bowl2", "food3": "bowl1"},
                     False, monkeypatch)
    assert not dict(st)["pair_together"] and not dict(st)["odd_one_separated"]
