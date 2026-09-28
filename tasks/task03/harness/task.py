"""Single-stage tabletop schedules, with no dependency on task01."""
import hashlib
import os

from .tabletop.budgets import INTERACTION_STEPS
from rlebench.runtime.capabilities import MOBILE_ACTIONS

TASKS = ("TowerMaxHeight", "CantileverOverhang", "BalanceCoins", "RubikCube", "HiddenCOM")


def configuration():
    task = os.environ["RLEBENCH_TASK"]
    if task not in TASKS:
        raise ValueError("unknown tabletop task")
    mode = "pocket" if task == "RubikCube" else "hidden_com" if task == "HiddenCOM" else "tabletop"
    budget = INTERACTION_STEPS["PocketCube" if mode == "pocket" else task]
    cameras = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    dimension, layout = 12, MOBILE_ACTIONS
    # Same deterministic seed used by the previous single-trial composition.
    seed = int.from_bytes(hashlib.sha256(f"rlebench.task01.eval||{task}|scene1|trial0".encode()).digest()[:8], "big") % (2**31-1)
    plan = [dict(task=task, seed=seed, mode=mode, default_resolution=512)]
    if mode == "pocket":
        cameras, dimension = ["front"], 14
        layout = {"robot0_arm_osc_pose": [0, 6], "robot0_gripper": [6, 7],
                  "robot1_arm_osc_pose": [7, 13], "robot1_gripper": [13, 14]}
    elif mode == "hidden_com":
        from .tabletop.hidden_com.config import CASES
        cameras, dimension = ["workspace", "closeup", "top"], 7
        layout = {"arm_osc_pose": [0, 6], "gripper": [6, 7]}
        plan = [dict(task=task, seed=case.seed, case=i, mode=mode, default_resolution=512)
                for i, case in enumerate(CASES)]
    return dict(mode=mode, adapter="harness.adapter", metrics="harness.metrics", train=[task], plan=plan,
                seconds=dict(evaluate=3600 if mode=="hidden_com" else 32400),
                budget=budget, horizon=budget,
                public=dict(action_dim=dimension, action_layout=layout, cameras=cameras,
                            depth=mode == "pocket", default_resolution=512, max_resolution=512))
