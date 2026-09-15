"""Public contract for fault-adaptive Panda control."""

from __future__ import annotations

import numpy as np

N_JOINTS = 7
HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
TORQUE_LIMIT = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
PHYSICS_DT = 0.002
CONTROL_DECIMATION = 5
CONTROL_DT = PHYSICS_DT * CONTROL_DECIMATION
CALIBRATION_STEPS = 160
EVALUATION_STEPS = 700
ACT_REPLY_SECONDS = 2.0
STARTUP_SECONDS = 30.0
MAX_SOURCE_BYTES = 1_048_576

FAULT_BOUNDS = {
    "encoder_zero_rad": [-0.14, 0.14],
    "actuator_gain": [0.72, 1.22],
    "weak_joint_multiplier": [0.62, 0.78],
    "viscous_friction": [0.0, 3.0],
    "command_delay_steps": [0, 3],
    "adjacent_crosstalk": [-0.11, 0.11],
    "payload_scale": [1.0, 2.6],
    "midrun_derating": [0.62, 0.78],
}


def public_context(model_file: str, scenario: str) -> dict:
    return {
        "model_file": model_file,
        "scenario": scenario,
        "joint_names": [f"joint{i}" for i in range(1, 8)],
        "actuator_names": [f"actuator{i}" for i in range(1, 8)],
        "ee_site": "attachment_site",
        "home": HOME.tolist(),
        "torque_limit": TORQUE_LIMIT.tolist(),
        "control_dt": CONTROL_DT,
        "calibration_steps": CALIBRATION_STEPS,
        "evaluation_steps": EVALUATION_STEPS,
        "fault_bounds": FAULT_BOUNDS,
        "action_contract": "seven normalized torques in [-1, 1]",
    }
