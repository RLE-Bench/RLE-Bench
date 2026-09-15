"""Deterministic phase-1 difficulty probe for fault-adaptive arm control.

This is a calibration experiment, not the task harness. It compares a nominal
controller, a conservative robust controller, and a privileged oracle ceiling on the
same hidden fault draws. The next phase replaces the oracle with an online estimator.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODEL_PATH = REPO / "assets/robots/franka_emika_panda/panda_nohand.xml"
HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
TORQUE_LIMIT = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
DT = 0.002
CONTROL_DECIMATION = 5
CONTROL_DT = DT * CONTROL_DECIMATION
HORIZON = 600
METRIC_WARMUP = 50
CALIBRATION_STEPS = 120
GATE = {
    "nominal_mean_max": 0.45,
    "adaptive_mean_min": 0.45,
    "adaptive_mean_max": 0.72,
    "oracle_mean_min": 0.90,
    "oracle_min_min": 0.85,
    "oracle_adaptive_gap_min": 0.25,
}
DERATE_STEP = HORIZON // 2
IMPULSE_START = HORIZON // 3
IMPULSE_STEPS = 12


@dataclass(frozen=True)
class Fault:
    zero_offset: tuple[float, ...]
    gain: tuple[float, ...]
    extra_friction: tuple[float, ...]
    delay_steps: int
    payload_scale: float
    weak_joint: int
    derate_factor: float
    crosstalk: tuple[tuple[float, ...], ...]
    impulse_joint: int
    impulse_torque: float


@dataclass(frozen=True)
class Result:
    seed: int
    controller: str
    rmse: float
    peak_error: float
    effort: float
    limit_violation: bool
    score: float


def sample_fault(seed: int) -> Fault:
    rng = np.random.default_rng(seed)
    gain = rng.uniform(0.72, 1.22, size=7)
    weak_joint = int(rng.integers(0, 7))
    gain[weak_joint] *= rng.uniform(0.62, 0.78)
    crosstalk = np.eye(7)
    for joint in range(6):
        crosstalk[joint, joint + 1] = rng.uniform(-0.11, 0.11)
        crosstalk[joint + 1, joint] = rng.uniform(-0.07, 0.07)
    return Fault(
        zero_offset=tuple(rng.uniform(-0.14, 0.14, size=7)),
        gain=tuple(gain),
        extra_friction=tuple(rng.uniform(0.0, 3.0, size=7)),
        delay_steps=int(rng.integers(0, 4)),
        payload_scale=float(rng.uniform(1.0, 2.6)),
        weak_joint=weak_joint,
        derate_factor=float(rng.uniform(0.62, 0.78)),
        crosstalk=tuple(tuple(float(value) for value in row) for row in crosstalk),
        impulse_joint=int(rng.integers(0, 4)),
        impulse_torque=float(rng.choice((-1.0, 1.0)) * rng.uniform(22.0, 38.0)),
    )


def build_model(payload_scale: float) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.timestep = DT
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.iterations = 50
    model.opt.ls_iterations = 20
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_gainprm[:, 0] = 1.0
    model.actuator_biasprm[:, :] = 0.0
    model.actuator_ctrlrange[:, 0] = -TORQUE_LIMIT
    model.actuator_ctrlrange[:, 1] = TORQUE_LIMIT
    model.actuator_forcerange[:, 0] = -TORQUE_LIMIT
    model.actuator_forcerange[:, 1] = TORQUE_LIMIT
    model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")] *= payload_scale
    return model


def reference(step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = step * CONTROL_DT
    frequency = np.array([0.11, 0.13, 0.09, 0.07, 0.15, 0.12, 0.1])
    amplitude = np.array([0.35, 0.28, 0.3, 0.24, 0.3, 0.25, 0.28])
    phase = np.zeros(7)
    omega = 2.0 * np.pi * frequency
    q = HOME + amplitude * np.sin(omega * t + phase)
    dq = amplitude * omega * np.cos(omega * t + phase)
    ddq = -amplitude * omega**2 * np.sin(omega * t + phase)
    return q, dq, ddq


def nominal_torque(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, dq: np.ndarray,
                   q_ref: np.ndarray, dq_ref: np.ndarray, ddq_ref: np.ndarray,
                   kp: float, kd: float) -> np.ndarray:
    data.qpos[:] = q
    data.qvel[:] = dq
    data.qacc[:] = ddq_ref
    mujoco.mj_inverse(model, data)
    return data.qfrc_inverse.copy() + kp * (q_ref - q) + kd * (dq_ref - dq)


def actuator_output(command: np.ndarray, gain: np.ndarray, crosstalk: np.ndarray,
                    friction: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current = crosstalk @ (gain * command)
    return current, np.clip(current - friction * velocity, -TORQUE_LIMIT, TORQUE_LIMIT)


def score_from_metrics(rmse: float, peak: float, effort: float, unsafe: bool) -> float:
    if unsafe:
        return 0.05
    tracking = np.clip((0.18 - rmse) / (0.18 - 0.025), 0.0, 1.0)
    peak_score = np.clip((0.5 - peak) / (0.5 - 0.12), 0.0, 1.0)
    effort_score = np.clip((0.8 - effort) / 0.45, 0.0, 1.0)
    return float(0.65 * tracking + 0.25 * peak_score + 0.10 * effort_score)


def run(seed: int, controller: str) -> Result:
    fault = sample_fault(seed)
    plant = build_model(fault.payload_scale)
    nominal = build_model(1.0)
    plant_data = mujoco.MjData(plant)
    nominal_data = mujoco.MjData(nominal)
    plant_data.qpos[:] = HOME
    mujoco.mj_forward(plant, plant_data)
    queue = deque([np.zeros(7) for _ in range(fault.delay_steps + 1)],
                  maxlen=fault.delay_steps + 1)
    errors: list[float] = []
    efforts: list[float] = []
    peak = 0.0
    unsafe = False
    gain = np.asarray(fault.gain)
    offset = np.asarray(fault.zero_offset)
    friction = np.asarray(fault.extra_friction)
    crosstalk = np.asarray(fault.crosstalk)
    sensor_rng = np.random.default_rng(seed + 1_000_003)
    oracle_data = mujoco.MjData(plant)
    predictor_data = mujoco.MjData(plant)

    offset_estimate = plant_data.qpos.copy() + offset - HOME
    calibration_commands: list[np.ndarray] = []
    calibration_currents: list[np.ndarray] = []
    probe_amplitude = np.array([8.0, 8.0, 7.0, 7.0, 2.0, 2.0, 2.0])
    for step in range(CALIBRATION_STEPS):
        q_true = plant_data.qpos.copy()
        dq_true = plant_data.qvel.copy()
        control_q = q_true if controller in ("adaptive", "oracle") else q_true + offset
        hold = nominal_torque(nominal, nominal_data, control_q, dq_true, HOME,
                              np.zeros(7), np.zeros(7), kp=60.0, kd=18.0)
        probe = probe_amplitude * np.sin(0.31 * step + np.arange(7) * 0.67)
        command = np.clip(hold + probe, -TORQUE_LIMIT, TORQUE_LIMIT)
        queue.append(command)
        motor_current, torque = actuator_output(queue[0], gain, crosstalk, friction, dq_true)
        observed_current = np.round(
            (motor_current + sensor_rng.normal(0.0, 0.18, 7)) / 0.25
        ) * 0.25
        plant_data.ctrl[:] = torque
        calibration_commands.append(command.copy())
        calibration_currents.append(observed_current)
        for _ in range(CONTROL_DECIMATION):
            mujoco.mj_step(plant, plant_data)

    commands = np.asarray(calibration_commands)
    currents = np.asarray(calibration_currents)
    delay_errors = []
    delay_gains = []
    for candidate in range(4):
        start = 5 + candidate
        x = commands[start - candidate:CALIBRATION_STEPS - candidate]
        y = currents[start:]
        estimate = np.sum(x * y, axis=0) / np.maximum(np.sum(x * x, axis=0), 1e-9)
        delay_gains.append(estimate)
        delay_errors.append(float(np.mean((y - x * estimate) ** 2)))
    delay_estimate = int(np.argmin(delay_errors))
    gain_estimate = np.clip(delay_gains[delay_estimate], 0.25, 1.5)
    friction_estimate = np.full(7, 1.25)

    for step in range(HORIZON):
        q_ref, dq_ref, ddq_ref = reference(step)
        q_true = plant_data.qpos.copy()
        dq_true = plant_data.qvel.copy()
        q_obs = q_true + offset
        effective_gain = gain.copy()
        if step >= DERATE_STEP:
            effective_gain[fault.weak_joint] *= fault.derate_factor

        if controller == "nominal":
            command = nominal_torque(nominal, nominal_data, q_obs, dq_true, q_ref, dq_ref,
                                     ddq_ref, kp=75.0, kd=16.0)
        elif controller == "robust":
            command = nominal_torque(nominal, nominal_data, q_obs, dq_true, q_ref, dq_ref,
                                     ddq_ref, kp=85.0, kd=25.0)
        elif controller == "adaptive":
            future_ref = reference(step + delay_estimate)
            corrected_q = q_obs - offset_estimate
            command = nominal_torque(nominal, nominal_data, corrected_q, dq_true,
                                     *future_ref, kp=80.0, kd=18.0)
            command = (command + friction_estimate * dq_true) / gain_estimate
        elif controller == "oracle":
            future_ref = reference(step + fault.delay_steps)
            corrected_q = q_obs - offset
            predictor_data.qpos[:] = corrected_q
            predictor_data.qvel[:] = dq_true
            mujoco.mj_forward(plant, predictor_data)
            for lookahead, pending_command in enumerate(list(queue)[1:]):
                predicted_gain = gain.copy()
                if step + lookahead >= DERATE_STEP:
                    predicted_gain[fault.weak_joint] *= fault.derate_factor
                _, predicted_torque = actuator_output(
                    pending_command, predicted_gain, crosstalk, friction,
                    predictor_data.qvel,
                )
                predictor_data.ctrl[:] = predicted_torque
                for _ in range(CONTROL_DECIMATION):
                    mujoco.mj_step(plant, predictor_data)
            desired = nominal_torque(
                plant, oracle_data, predictor_data.qpos, predictor_data.qvel,
                *future_ref, kp=80.0, kd=18.0,
            )
            command = np.linalg.solve(
                crosstalk, desired + friction * predictor_data.qvel,
            )
            command = command / effective_gain
        else:
            raise ValueError(controller)

        command = np.clip(command, -TORQUE_LIMIT, TORQUE_LIMIT)
        queue.append(command)
        delayed = queue[0]
        _, actual = actuator_output(delayed, effective_gain, crosstalk, friction, dq_true)
        if IMPULSE_START <= step < IMPULSE_START + IMPULSE_STEPS:
            actual[fault.impulse_joint] += fault.impulse_torque
        plant_data.ctrl[:] = np.clip(actual, -TORQUE_LIMIT, TORQUE_LIMIT)
        for _ in range(CONTROL_DECIMATION):
            mujoco.mj_step(plant, plant_data)

        if step >= METRIC_WARMUP:
            error = float(np.sqrt(np.mean((plant_data.qpos - q_ref) ** 2)))
            errors.append(error)
            peak = max(peak, error)
            efforts.append(float(np.mean(np.abs(command) / TORQUE_LIMIT)))
        lower, upper = plant.jnt_range[:, 0], plant.jnt_range[:, 1]
        if np.any(plant_data.qpos < lower - 0.03) or np.any(plant_data.qpos > upper + 0.03):
            unsafe = True
            break

    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("inf")
    effort = float(np.mean(efforts)) if efforts else 1.0
    return Result(seed, controller, rmse, peak, effort, unsafe,
                  score_from_metrics(rmse, peak, effort, unsafe))


def evaluate_gate(summary: dict[str, dict[str, float | int]]) -> dict[str, bool]:
    nominal = float(summary["nominal"]["score_mean"])
    adaptive = float(summary["adaptive"]["score_mean"])
    oracle = float(summary["oracle"]["score_mean"])
    return {
        "nominal_is_hard": nominal <= GATE["nominal_mean_max"],
        "adaptive_has_signal": adaptive >= GATE["adaptive_mean_min"],
        "adaptive_not_solved": adaptive <= GATE["adaptive_mean_max"],
        "oracle_is_solvable": oracle >= GATE["oracle_mean_min"],
        "oracle_tail_is_solvable": float(summary["oracle"]["score_min"]) >= GATE["oracle_min_min"],
        "identification_gap": oracle - adaptive >= GATE["oracle_adaptive_gap_min"],
        "oracle_is_safe": int(summary["oracle"]["unsafe"]) == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    controllers = ("nominal", "robust", "adaptive", "oracle")
    results = [run(seed, controller) for controller in controllers
               for seed in range(args.seeds)]
    summary: dict[str, dict[str, float | int]] = {}
    for controller in controllers:
        group = [r for r in results if r.controller == controller]
        summary[controller] = {
            "score_mean": float(np.mean([r.score for r in group])),
            "score_min": float(np.min([r.score for r in group])),
            "score_max": float(np.max([r.score for r in group])),
            "rmse_mean": float(np.mean([r.rmse for r in group])),
            "unsafe": int(sum(r.limit_violation for r in group)),
        }
    checks = evaluate_gate(summary)
    payload = {
        "config": {"seeds": args.seeds, "horizon": HORIZON, "gate": GATE},
        "summary": summary,
        "difficulty_checks": checks,
        "difficulty_gate_passed": all(checks.values()),
        "results": [asdict(r) for r in results],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        args.json.write_text(rendered + "\n")
    return 1 if args.check and not all(checks.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
