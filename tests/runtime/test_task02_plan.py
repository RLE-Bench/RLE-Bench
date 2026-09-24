import os
import pytest
if os.environ.get("RLEBENCH_TEST_TASK") != "task02":
    pytest.skip("family-specific checks", allow_module_level=True)

import importlib
from harness import config as C

def test_the_shipped_split_satisfies_its_own_rule():
    """One activity group, its median-subtask member held out, the rest trained on."""
    C.check_split()


def test_the_split_names_its_activity_group():
    assert C.ACTIVITY_GROUP


def test_check_split_rejects_more_than_one_evaluation_task():
    with pytest.raises(ValueError, match="exactly one evaluation task"):
        C.check_split(train=("A",), evaluation=("B", "C"))


def test_check_split_rejects_an_overlapping_split():
    with pytest.raises(ValueError, match="overlap"):
        C.check_split(train=("A",), evaluation=("A",))


def test_both_splits_are_real_composite_tasks():
    """The atomic/composite boundary still matters: an 'activity member' that was
    secretly atomic would make the evaluation a primitive, not a composition."""
    pytest.importorskip("robocasa", reason="RoboCasa lives in the task venv")
    C.check_registry()


def test_check_split_rejects_a_wrongly_sized_training_set():
    with pytest.raises(ValueError, match="expected"):
        C.check_split(train=tuple("ABCDEFG"[:C.TRAIN_SET_SIZE + 1]), evaluation=("Z",))


def test_check_split_rejects_a_repeated_training_task():
    train = ("A",) * C.TRAIN_SET_SIZE
    with pytest.raises(ValueError, match="repeats"):
        C.check_split(train=train, evaluation=("Z",))


def test_the_splits_are_fixed_constants_not_environment_reads(monkeypatch):
    """They were environment variables once, which put the graded task names in the
    container's config, where `docker exec` hands them to the agent whatever the
    entrypoint does. A split that cannot be set at run time cannot leak at run time."""
    import importlib

    monkeypatch.setenv("RLEBENCH_TRAIN_TASKS", "OpenDrawer")
    monkeypatch.setenv("RLEBENCH_EVAL_TASKS", "Nonsense")
    reloaded = importlib.reload(C)
    try:
        assert reloaded.TRAIN_TASKS != ("OpenDrawer",)
        assert "Nonsense" not in reloaded.EVAL_TASKS
    finally:
        importlib.reload(C)


def test_plan_is_deterministic():
    assert C.eval_plan() == C.eval_plan()


def test_plan_is_trials_per_task_trials_per_task():
    """One trial is one Harbor step is one fresh agent, and each task gets
    TRIALS_PER_TASK of them -- repeated draws on one held-out task are what distinguish a
    harness that transfers from one that got lucky once."""
    plan = C.eval_plan()
    assert len(plan) == len(C.EVAL_TASKS) * C.TRIALS_PER_TASK
    for task in C.EVAL_TASKS:
        assert [t for t, _ in plan].count(task) == C.TRIALS_PER_TASK


def test_the_trial_count_is_not_settable_from_the_environment(monkeypatch):
    """It is the length of a STATIC [[steps]] list, so a run-time override would desync
    the two with nothing to say so. Change it with build_groups.py --trials."""
    monkeypatch.setenv("RLEBENCH_TRIALS_PER_TASK", "99")
    importlib.reload(C)
    try:
        assert C.TRIALS_PER_TASK != 99
    finally:
        monkeypatch.delenv("RLEBENCH_TRIALS_PER_TASK")
        importlib.reload(C)


def test_trial_seeds_are_unique():
    seeds = [seed for _, seed in C.eval_plan()]
    assert len(set(seeds)) == len(seeds)


def test_a_salt_changes_the_plan():
    """The salt is what lets a deployment keep its trials private even if the code is."""
    assert C.eval_plan() != C.eval_plan(salt="secret")


def test_seeds_are_in_a_valid_range():
    for _, seed in C.eval_plan():
        assert 0 <= seed < 2**31 - 1


def test_narrowing_the_trial_count_does_not_renumber_the_remaining_trials():
    """A reduced-scale run must grade the same episodes as the full one, or a smoke
    run tells you nothing about the real one. Seeds are keyed on position in the WHOLE
    plan, so narrowing yields a prefix rather than a fresh numbering."""
    full = C.eval_plan()
    assert C.eval_plan(trials_per_task=2) == full[:2]


def test_narrowing_the_task_list_does_not_renumber_the_remaining_trials():
    """The same property along the other axis, which is what matters once a scale-up
    grades more than one held-out task."""
    tasks = ("Alpha", "Beta", "Gamma")
    full = C.eval_plan(tasks=tasks, trials_per_task=3)
    assert C.eval_plan(tasks=tasks[:2], trials_per_task=3) == full[:6]


def test_a_trial_count_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        C.eval_plan(trials_per_task=0)


def test_an_empty_split_is_refused():
    with pytest.raises(ValueError, match="at least one task"):
        C.eval_plan(tasks=())


def test_object_state_never_reaches_the_agent():
    """A composite success predicate is a conjunction over named objects, so an
    unfiltered observation would hand over the exact quantities being scored."""
    obs = {
        "robot0_eef_pos": 1, "robot0_agentview_left_image": 2,
        "robot0_proprio-state": 3,
        "fruit_pos": 4, "object-state": 5, "obj_to_robot0_eef_pos": 6,
    }
    visible = C.agent_visible_obs(obs)
    assert set(visible) == {"robot0_eef_pos", "robot0_agentview_left_image",
                            "robot0_proprio-state"}


def test_non_dict_observations_pass_through():
    assert C.agent_visible_obs(None) is None


def test_budgets_are_positive():
    assert C.INTERACTION_STEPS > 0
    assert C.MAX_STEPS_PER_TRIAL > 0
    assert C.MAX_EPISODE_STEPS > 0
    assert C.ENV_CACHE_SIZE >= 1


def test_observe_may_ask_for_more_than_the_step_pipeline_renders():
    assert C.OBS_MAX_RESOLUTION >= C.OBS_RESOLUTION


def test_evaluation_runs_on_robocasas_held_out_split():
    """Not a knob. Development on "pretrain", evaluation on "target", is RoboCasa's
    published test protocol."""
    assert (C.DEV_SPLIT, C.EVAL_SPLIT) == ("pretrain", "target")
