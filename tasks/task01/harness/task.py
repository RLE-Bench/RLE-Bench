"""Private speed-run schedule and public capabilities."""
import os
from . import config as C
from rlebench.runtime.capabilities import MOBILE_ACTIONS

CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]


def configuration():
    task = os.environ["RLEBENCH_TASK"]
    level = os.environ.get("RLEBENCH_LEVEL", "L1")
    if level != os.environ.get("RLEBENCH_IMAGE_LEVEL", level) or level not in C.LEVELS:
        raise ValueError("image and task level mismatch")
    if task not in [name for name, _ in C.SUBTASKS] or not C.is_scoreable(task):
        raise ValueError("unknown task")
    overrides = C.eval_overrides()
    plan = C.parse_eval_plan(overrides["plan"]) if overrides.get("plan") else C.EVAL_PLAN
    trials = [dict(task=task, seed=seed, scene=[layout, style], split="target",
                   level=level, default_resolution=C.OBS_RESOLUTION, mode="task01")
              for _, layout, style, seed in C.eval_trials(task, plan, overrides.get("salt"))]
    return dict(mode="task01", adapter="harness.adapter", train=[task], plan=trials, level=level,
                seconds=dict(session=C.env_float("RLEBENCH_SESSION_SECONDS",32400)),
                budget=C.env_int("RLEBENCH_INTERACTION_STEPS", C.INTERACTION_STEPS),
                horizon=C.env_int("RLEBENCH_MAX_STEPS_PER_TRIAL", C.MAX_STEPS_PER_TRIAL),
                public=dict(action_dim=12, cameras=CAMERAS, depth=True, default_resolution=C.OBS_RESOLUTION,
                            max_resolution=512, level=level, action_layout=MOBILE_ACTIONS))
