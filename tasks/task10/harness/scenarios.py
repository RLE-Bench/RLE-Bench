"""Public target functions and verifier-owned disturbance timing."""

from __future__ import annotations

import numpy as np

from . import spec


def target(scenario: str, step: int) -> dict:
    trajectory_step = max(step - 80, 0)
    t = trajectory_step * spec.CONTROL_DT
    if scenario == "free_space":
        position = np.array([0.54 + 0.045 * np.sin(0.9 * t), 0.075 * np.sin(1.3 * t), 0.61 + 0.035 * np.sin(0.7 * t)])
        force = 0.0
    elif scenario == "contact":
        u = min(trajectory_step / 430.0, 1.0)
        ramp = u * u * (3.0 - 2.0 * u)
        position = np.array([0.54 + 0.05 * np.sin(0.65 * t), 0.08 * np.sin(0.9 * t), 0.624 - 0.264 * ramp])
        force = 10.0 * min(max(trajectory_step - 410, 0) / 100.0, 1.0)
    elif scenario == "recovery":
        position = np.array([0.53 + 0.055 * np.sin(1.1 * t), 0.065 * np.sin(0.8 * t), 0.59 + 0.03 * np.cos(0.75 * t)])
        force = 0.0
    else:
        raise ValueError(scenario)
    return {"position": position, "normal_force": force, "phase": "calibration" if step < 0 else "evaluation"}


def disturbance(scenario: str, step: int) -> tuple[bool, bool]:
    impulse = scenario == "recovery" and 250 <= step < 262
    derated = scenario == "recovery" and step >= 400
    return impulse, derated
