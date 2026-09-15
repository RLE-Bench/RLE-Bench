"""Verifier-owned deterministic scenario evaluator."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from . import config, metrics, model as model_lib, scenarios, spec
from .faults import sample_fault


@dataclass
class EpisodeResult:
    seed: int
    scenario: str
    score: float
    position_rmse: float
    force_rmse: float
    peak_force: float
    recovery_seconds: float
    unsafe: bool
    action_faults: int


def _contact_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    probe, surface = model.geom("probe").id, model.geom("surface").id
    total = 0.0
    wrench = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        if {int(contact.geom1), int(contact.geom2)} == {probe, surface}:
            mujoco.mj_contactForce(model, data, index, wrench)
            total += abs(float(wrench[0]))
    return total


def evaluate_episode(controller, seed: int, scenario: str) -> EpisodeResult:
    fault = sample_fault(seed)
    model = model_lib.build_model(fault.payload_scale)
    data = mujoco.MjData(model)
    data.qpos[:] = spec.HOME
    mujoco.mj_forward(model, data)
    gain, friction = np.asarray(fault.gain), np.asarray(fault.friction)
    coupling, offset = np.asarray(fault.crosstalk), np.asarray(fault.zero_offset)
    queue = deque([np.zeros(7) for _ in range(fault.delay_steps + 1)], maxlen=fault.delay_steps + 1)
    rng = np.random.default_rng(seed + 2_000_033)
    command = np.zeros(7)
    action_faults = 0
    position_errors, force_errors, forces = [], [], []
    recovery = config.RECOVERY_FAIL_S
    recovered_ticks = 0
    filtered_force = 0.0
    force_over_ticks = 0
    unsafe_force = False
    joint_unsafe = False
    total_steps = spec.CALIBRATION_STEPS + spec.EVALUATION_STEPS
    for clock in range(total_steps):
        evaluation_step = clock - spec.CALIBRATION_STEPS
        if evaluation_step == 0:
            data.qpos[:] = spec.HOME
            data.qvel[:] = 0.0
            data.ctrl[:] = 0.0
            queue.clear()
            queue.extend(np.zeros(7) for _ in range(fault.delay_steps + 1))
            command[:] = 0.0
            mujoco.mj_forward(model, data)
        target = scenarios.target(scenario, evaluation_step)
        if evaluation_step < 0:
            target = {"position": data.site("attachment_site").xpos.copy(), "normal_force": 0.0, "phase": "calibration"}
        raw_force = _contact_force(model, data)
        filtered_force = 0.95 * filtered_force + 0.05 * raw_force
        force = filtered_force
        observation = {
            "step": clock,
            "qpos": data.qpos.copy() + offset,
            "qvel": data.qvel.copy(),
            "commanded_torque": command.copy(),
            "motor_current": np.round((coupling @ (gain * queue[0]) + rng.normal(0.0, 0.18, 7)) / 0.25) * 0.25,
            "ee_position": data.site("attachment_site").xpos.copy(),
            "contact_force": force,
            "target": {"position": np.asarray(target["position"]).tolist(), "normal_force": target["normal_force"], "phase": target["phase"]},
        }
        try:
            normalized = np.asarray(controller.act(observation), dtype=float)
            if normalized.shape != (7,) or not np.isfinite(normalized).all():
                raise ValueError("bad action")
            command = np.clip(normalized, -1.0, 1.0) * spec.TORQUE_LIMIT
        except Exception:
            action_faults += 1
            command = np.zeros(7)
        queue.append(command)
        effective_gain = gain.copy()
        impulse, derated = scenarios.disturbance(scenario, evaluation_step)
        if derated:
            effective_gain[fault.weak_joint] *= fault.derate_factor
        current = coupling @ (effective_gain * queue[0])
        applied = current - friction * data.qvel
        if impulse:
            applied[fault.impulse_joint] += fault.impulse_torque
        data.ctrl[:] = np.clip(applied, -spec.TORQUE_LIMIT, spec.TORQUE_LIMIT)
        for _ in range(spec.CONTROL_DECIMATION):
            mujoco.mj_step(model, data)
        lower, upper = model.jnt_range[:, 0], model.jnt_range[:, 1]
        joint_unsafe = joint_unsafe or bool(
            np.any(data.qpos < lower - config.JOINT_LIMIT_MARGIN)
            or np.any(data.qpos > upper + config.JOINT_LIMIT_MARGIN)
        )
        if evaluation_step >= 40:
            error = float(np.linalg.norm(data.site("attachment_site").xpos - np.asarray(target["position"])))
            position_errors.append(error)
            if scenario == "contact" and float(target["normal_force"]) >= 1.0:
                force_errors.append(force - float(target["normal_force"]))
                forces.append(force)
                force_over_ticks = force_over_ticks + 1 if force > config.MAX_CONTACT_FORCE_N else 0
                unsafe_force = unsafe_force or force_over_ticks >= 5
            if scenario == "recovery" and evaluation_step >= 262:
                recovered_ticks = recovered_ticks + 1 if error < 0.045 else 0
                if recovered_ticks == 20 and recovery == config.RECOVERY_FAIL_S:
                    recovery = (evaluation_step - 262 - 19) * spec.CONTROL_DT
    lower, upper = model.jnt_range[:, 0], model.jnt_range[:, 1]
    unsafe = bool(np.any(data.qpos < lower - config.JOINT_LIMIT_MARGIN) or np.any(data.qpos > upper + config.JOINT_LIMIT_MARGIN) or unsafe_force or action_faults > config.MAX_ACTION_FAULTS)
    position_rmse = float(np.sqrt(np.mean(np.square(position_errors))))
    force_rmse = float(np.sqrt(np.mean(np.square(force_errors)))) if force_errors else 0.0
    peak_force = max(forces, default=0.0)
    if scenario == "free_space":
        score = metrics.tracking_score(position_rmse)
    elif scenario == "contact":
        score = metrics.contact_score(position_rmse, force_rmse, peak_force)
        if sum(force > 2.0 for force in forces) < 30:
            score = 0.0
    else:
        score = metrics.recovery_score(position_rmse, recovery)
    if unsafe:
        score = min(score, config.GATE_CAP)
    return EpisodeResult(seed, scenario, score, position_rmse, force_rmse, peak_force, recovery, unsafe, action_faults)


def evaluate_factory(factory, seeds=config.EVAL_SEEDS) -> dict:
    results = []
    for scenario in config.SCENARIOS:
        for seed in seeds:
            with TemporaryDirectory() as directory:
                nominal = f"{directory}/nominal.mjb"
                model_lib.save_nominal(nominal)
                controller = factory()
                controller.reset(spec.public_context(nominal, scenario), 0)
                results.append(evaluate_episode(controller, seed, scenario))
    bands = {scenario: float(np.mean([result.score for result in results if result.scenario == scenario])) for scenario in config.SCENARIOS}
    reward = sum(config.WEIGHTS[name] * bands[name] for name in config.SCENARIOS)
    if any(result.unsafe for result in results):
        reward = min(reward, config.GATE_CAP)
    return {"reward": float(np.clip(reward, 0.0, 1.0)), "bands": bands, "episodes": [asdict(result) for result in results]}
