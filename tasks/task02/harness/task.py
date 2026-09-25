"""Private transfer split; only training members are exposed to the client."""
from . import config as C
from rlebench.runtime.capabilities import MOBILE_ACTIONS


def configuration():
    C.check_split()
    return dict(mode="task02", adapter="harness.adapter", train=list(C.TRAIN_TASKS),
                seconds=dict(develop=C.env_float("RLEBENCH_DEVELOP_SECONDS",28800), evaluate=C.env_float("RLEBENCH_TRIAL_SECONDS",3600)),
                plan=[dict(task=task, seed=seed, split="target", level="L1", mode="task02",
                           default_resolution=C.OBS_RESOLUTION) for task, seed in C.eval_plan()],
                budget=C.env_int("RLEBENCH_INTERACTION_STEPS", C.INTERACTION_STEPS),
                horizon=C.env_int("RLEBENCH_MAX_STEPS_PER_TRIAL", C.MAX_STEPS_PER_TRIAL),
                public=dict(action_dim=12, cameras=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
                            depth=True, default_resolution=C.OBS_RESOLUTION, max_resolution=512,
                            action_layout=MOBILE_ACTIONS))
