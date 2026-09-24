import os
import pytest
if os.environ.get("RLEBENCH_TEST_TASK") != "task01":
    pytest.skip("family-specific checks", allow_module_level=True)

from harness import config as C

def test_plan_is_deterministic():
    assert C.eval_trials("CloseDrawer") == C.eval_trials("CloseDrawer")


def test_plan_differs_between_tasks():
    """Otherwise every task would be graded on the same episodes."""
    assert C.eval_trials("CloseDrawer") != C.eval_trials("OpenDrawer")


def test_trial_seeds_are_unique_within_a_plan():
    """A repeated seed would silently make the evaluation smaller than it claims."""
    plan = C.eval_trials("CloseDrawer", plan=((1, 8), (2, 8)))
    seeds = [p[3] for p in plan]
    assert len(set(seeds)) == len(seeds) == 16


def test_a_salt_changes_the_plan():
    """The salt is what lets a deployment keep its trials private even if the code
    is public."""
    assert C.eval_trials("CloseDrawer") != C.eval_trials("CloseDrawer", salt="secret")


def test_adding_trials_does_not_renumber_the_existing_ones():
    """Growing a scene's trial count must not change the trials already in it, or
    results from different plan sizes would not be comparable."""
    small = C.eval_trials("CloseDrawer", plan=((1, 2),))
    large = C.eval_trials("CloseDrawer", plan=((1, 5),))
    assert large[:2] == small


def test_seeds_are_in_a_valid_range():
    for entry in C.eval_trials("CloseDrawer", plan=((1, 16),)):
        assert 0 <= entry[3] < 2**31 - 1


def test_scene_ids_map_to_target_split_pairs():
    """Scene ids are 1-based into the target split's own (layout, style) pairs, so
    pinning a scene stays inside RoboCasa's distribution."""
    plan = C.eval_trials("CloseDrawer", plan=((1, 1), (4, 1)))
    assert [(p[1], p[2]) for p in plan] == [C.TARGET_SCENE_IDS[0],
                                            C.TARGET_SCENE_IDS[3]]
    # A scene's trials come out together and in plan order, so a reader can map a
    # trial index back to the scene it ran in.
    wider = C.eval_trials("CloseDrawer", plan=((1, 2), (3, 1)))
    assert [p[0] for p in wider] == [1, 1, 3]


def test_plan_rejects_impossible_schedules():
    for bad in ((), ((1, 0),), ((0, 1),), ((99, 1),)):
        with pytest.raises(ValueError):
            C.eval_trials("CloseDrawer", plan=bad)


def test_the_reset_satisfied_task_is_excluded():
    """PrepareBroilingStation's predicate is true at reset, so a do-nothing agent
    scores 100% -- it measures nothing and must not be scoreable."""
    assert not C.is_scoreable("PrepareBroilingStation")
    assert C.is_scoreable("CloseDrawer")


def test_recorded_working_assumptions():
    """Recorded so a change is a deliberate edit here, not drift elsewhere."""
    # One terminal submission: the score is final, there is no interact/evaluate loop.
    assert C.SUBMISSIONS == 1
    assert C.MAX_STEPS_PER_TRIAL > 0
    # The two reward weights are the whole formula and must sum to 1.0, or a perfect
    # free run would not score 1.0 and scores would stop being comparable across
    # deployments that tuned them.
    assert C.W_OUTCOME + C.W_EFFICIENCY == pytest.approx(1.0)
    assert C.W_OUTCOME > C.W_EFFICIENCY, "solving the task must dominate"


def test_the_default_plan_can_resolve_a_success_rate(tmp_path):
    """One trial makes the score binary, which cannot rank anything between total
    failure and total success. The default must be wide enough to have a middle."""
    assert C.plan_trial_count() >= 5
    scenes = {scene for scene, _ in C.EVAL_PLAN}
    assert len(scenes) > 1, "a single scene measures luck, not generalisation"


def test_plan_spec_round_trips():
    assert C.parse_eval_plan("1x2,2x2,3x2") == ((1, 2), (2, 2), (3, 2))
    assert C.parse_eval_plan(" 4x1 ") == ((4, 1),)


@pytest.mark.parametrize("spec", [
    "",                # empty
    "1",               # no trial count
    "1x",              # no trial count
    "axb",             # not integers
    "1x0",             # a scene with no trials
    "0x2",             # scene ids are 1-based
    "99x2",            # beyond the target split's scenes
])
def test_a_malformed_plan_is_refused(spec):
    """The daemon parses this at startup so a bad plan fails where it is visible,
    rather than at the first graded trial."""
    with pytest.raises(ValueError):
        C.parse_eval_plan(spec)


def test_plan_trial_count_follows_the_plan():
    assert C.plan_trial_count(C.parse_eval_plan("1x3,2x4")) == 7


def test_env_overrides_are_read_and_validated(monkeypatch):
    monkeypatch.setenv("RLEBENCH_INTERACTION_STEPS", "250")
    assert C.env_int("RLEBENCH_INTERACTION_STEPS", C.INTERACTION_STEPS) == 250
    monkeypatch.setenv("RLEBENCH_INTERACTION_STEPS", "lots")
    with pytest.raises(ValueError):
        C.env_int("RLEBENCH_INTERACTION_STEPS", C.INTERACTION_STEPS)
    monkeypatch.delenv("RLEBENCH_INTERACTION_STEPS")
    assert C.env_int("RLEBENCH_INTERACTION_STEPS", 7) == 7
