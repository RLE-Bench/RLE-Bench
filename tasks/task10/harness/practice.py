"""Agent-visible runner on explicit public fault profiles; no scoring code."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import deque
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from . import model as model_lib
from . import scenarios, spec

PROFILES = {
    "mild": {"offset": [0.02, -0.01, 0.0, 0.03, 0.0, -0.02, 0.01], "gain": [0.92, 1.06, 0.88, 1.08, 0.95, 1.02, 0.9], "friction": [0.5] * 7, "delay": 1, "payload": 1.3, "weak": 3, "derate": 0.8, "impulse_joint": 1, "impulse": 20.0},
    "coupled": {"offset": [-0.08, 0.06, -0.04, 0.09, -0.05, 0.03, 0.07], "gain": [0.78, 1.12, 0.84, 0.72, 1.1, 0.9, 0.76], "friction": [1.7, 0.8, 2.1, 1.2, 2.3, 0.6, 1.5], "delay": 2, "payload": 2.0, "weak": 2, "derate": 0.7, "impulse_joint": 3, "impulse": -30.0},
    "stress": {"offset": [0.12, -0.11, 0.09, -0.1, 0.08, -0.06, 0.1], "gain": [0.75, 0.82, 0.68, 1.16, 0.74, 0.9, 0.8], "friction": [2.6, 2.0, 1.4, 2.8, 1.8, 2.4, 1.0], "delay": 3, "payload": 2.5, "weak": 0, "derate": 0.65, "impulse_joint": 2, "impulse": 36.0},
}


def contact_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    probe, surface = model.geom("probe").id, model.geom("surface").id
    wrench = np.zeros(6)
    total = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        if {int(contact.geom1), int(contact.geom2)} == {probe, surface}:
            mujoco.mj_contactForce(model, data, index, wrench)
            total += abs(float(wrench[0]))
    return total


def load_controller(path: str):
    module_spec = importlib.util.spec_from_file_location("practice_controller", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.make_controller()


def run(controller_file: str, profile_name: str, scenario: str) -> dict:
    profile = PROFILES[profile_name]
    plant = model_lib.build_model(profile["payload"])
    data = mujoco.MjData(plant)
    data.qpos[:] = spec.HOME
    mujoco.mj_forward(plant, data)
    offset, gain, friction = (np.asarray(profile[key], dtype=float) for key in ("offset", "gain", "friction"))
    coupling = np.eye(7)
    coupling[np.arange(6), np.arange(1, 7)] = 0.06
    queue = deque([np.zeros(7) for _ in range(profile["delay"] + 1)], maxlen=profile["delay"] + 1)
    command = np.zeros(7)
    errors = []
    force_errors = []
    forces = []
    filtered_force = 0.0
    with TemporaryDirectory() as directory:
        nominal = f"{directory}/nominal.mjb"
        model_lib.save_nominal(nominal)
        controller = load_controller(controller_file)
        controller.reset(spec.public_context(nominal, scenario), 12345)
        for clock in range(spec.CALIBRATION_STEPS + spec.EVALUATION_STEPS):
            step = clock - spec.CALIBRATION_STEPS
            if step == 0:
                data.qpos[:] = spec.HOME
                data.qvel[:] = 0.0
                queue.clear()
                queue.extend(np.zeros(7) for _ in range(profile["delay"] + 1))
                mujoco.mj_forward(plant, data)
            target = scenarios.target(scenario, step)
            if step < 0:
                target = {"position": data.site("attachment_site").xpos.copy(), "normal_force": 0.0, "phase": "calibration"}
            filtered_force = 0.95 * filtered_force + 0.05 * contact_force(plant, data)
            observation = {"step": clock, "qpos": data.qpos.copy() + offset, "qvel": data.qvel.copy(), "commanded_torque": command.copy(), "motor_current": coupling @ (gain * queue[0]), "ee_position": data.site("attachment_site").xpos.copy(), "contact_force": filtered_force, "target": {"position": np.asarray(target["position"]).tolist(), "normal_force": target["normal_force"], "phase": target["phase"]}}
            action = np.asarray(controller.act(observation), dtype=float)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("controller returned an invalid action")
            command = np.clip(action, -1.0, 1.0) * spec.TORQUE_LIMIT
            queue.append(command)
            effective_gain = gain.copy()
            if scenario == "recovery" and step >= 400:
                effective_gain[profile["weak"]] *= profile["derate"]
            applied = coupling @ (effective_gain * queue[0]) - friction * data.qvel
            if scenario == "recovery" and 250 <= step < 262:
                applied[profile["impulse_joint"]] += profile["impulse"]
            data.ctrl[:] = np.clip(applied, -spec.TORQUE_LIMIT, spec.TORQUE_LIMIT)
            for _ in range(spec.CONTROL_DECIMATION):
                mujoco.mj_step(plant, data)
            if step >= 40:
                errors.append(float(np.linalg.norm(data.site("attachment_site").xpos - np.asarray(target["position"]))))
                if scenario == "contact" and float(target["normal_force"]) >= 1.0:
                    force_errors.append(filtered_force - float(target["normal_force"]))
                    forces.append(filtered_force)
    return {"profile": profile_name, "scenario": scenario, "position_rmse": float(np.sqrt(np.mean(np.square(errors)))), "force_rmse": float(np.sqrt(np.mean(np.square(force_errors)))) if force_errors else 0.0, "peak_force": max(forces, default=0.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("controller")
    parser.add_argument("--profile", choices=PROFILES, default="mild")
    parser.add_argument("--scenario", choices=("free_space", "contact", "recovery"), default="free_space")
    args = parser.parse_args()
    print(json.dumps(run(args.controller, args.profile, args.scenario), indent=2))


if __name__ == "__main__":
    main()
