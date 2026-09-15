"""Submission scoring orchestration."""

from __future__ import annotations


from . import config
from .controller_process import ControllerFault, ControllerProcess
from .evaluator import evaluate_episode


def score_submission(controller_file: str) -> dict:
    episodes = []
    load_failures = 0
    for scenario in config.SCENARIOS:
        for seed in config.EVAL_SEEDS:
            try:
                with ControllerProcess(controller_file, scenario, 0) as process:
                    episodes.append(evaluate_episode(process, seed, scenario))
            except (ControllerFault, OSError, ValueError):
                load_failures += 1
    if not episodes:
        return {"reward": 0.0, "bands": {}, "episodes": [], "load_failures": load_failures}
    bands = {scenario: sum(result.score for result in episodes if result.scenario == scenario) / len([result for result in episodes if result.scenario == scenario]) for scenario in config.SCENARIOS}
    reward = sum(config.WEIGHTS[name] * bands.get(name, 0.0) for name in config.SCENARIOS)
    if load_failures or any(result.unsafe for result in episodes):
        reward = min(reward, config.GATE_CAP)
    return {"reward": float(reward), "bands": bands, "episodes": [result.__dict__ for result in episodes], "load_failures": load_failures}
