"""Observation-only reference controller; never shipped to the agent image."""

from __future__ import annotations

import mujoco
import numpy as np


class GoldenController:
    def reset(self, context: dict, seed: int) -> None:
        self.model = mujoco.MjModel.from_binary_path(context["model_file"])
        self.data = mujoco.MjData(self.model)
        self.home = np.asarray(context["home"], dtype=float)
        self.limits = np.asarray(context["torque_limit"], dtype=float)
        self.dt = float(context["control_dt"])
        self.site = self.model.site(context["ee_site"]).id
        self.scenario = context["scenario"]
        self.offset = None
        self.commands: list[np.ndarray] = []
        self.currents: list[np.ndarray] = []
        self.delay = 0
        self.map = np.eye(7)
        self.step = 0
        self.evaluation_step = 0
        self.bias_correction = np.zeros(7)
        self.contact_offset = 0.0
        self.probe_frequency = np.array([0.17, 0.23, 0.31, 0.41, 0.53, 0.67, 0.83])
        self.probe_phase = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, 7)

    def _fit(self) -> None:
        if len(self.commands) < 40:
            return
        commands, currents = np.asarray(self.commands[-150:]), np.asarray(self.currents[-150:])
        best = None
        for delay in range(4):
            x, y = commands[:len(commands) - delay or None], currents[delay:]
            if len(x) < 20:
                continue
            matrix = np.linalg.lstsq(x, y, rcond=2e-3)[0].T
            error = float(np.mean((y - x @ matrix.T) ** 2))
            if best is None or error < best[0]:
                best = error, delay, matrix
        if best is not None:
            _, self.delay, matrix = best
            self.map = 0.65 * self.map + 0.35 * matrix

    def act(self, observation: dict):
        q_obs = np.asarray(observation["qpos"], dtype=float)
        dq = np.asarray(observation["qvel"], dtype=float)
        if self.offset is None:
            self.offset = q_obs - self.home
        q = q_obs - self.offset
        previous = np.asarray(observation["commanded_torque"], dtype=float)
        current = np.asarray(observation["motor_current"], dtype=float)
        self.commands.append(previous)
        self.currents.append(current)
        phase = observation["target"]["phase"]
        if phase == "calibration":
            self.data.qpos[:] = q
            self.data.qvel[:] = dq
            self.data.qacc[:] = 0.0
            mujoco.mj_inverse(self.model, self.data)
            amplitude = np.array([3.0, 3.0, 2.5, 2.5, 0.8, 0.8, 0.8])
            torque = self.data.qfrc_inverse.copy()
            torque += amplitude * np.sin(self.probe_frequency * self.step + self.probe_phase)
            self.bias_correction += 12.0 * (self.home - q) * self.dt
            self.bias_correction = np.clip(self.bias_correction, -0.35 * self.limits, 0.35 * self.limits)
            torque += self.bias_correction - 18.0 * (q - self.home) - 10.0 * dq
            if self.step >= 99 and self.step % 20 == 19:
                self._fit()
        else:
            self.data.qpos[:] = q
            self.data.qvel[:] = dq
            if self.evaluation_step < 80:
                self.data.qacc[:] = 0.0
                mujoco.mj_inverse(self.model, self.data)
                torque = self.data.qfrc_inverse.copy() + self.bias_correction
                torque += -18.0 * (q - self.home) - 10.0 * dq
                try:
                    torque = np.linalg.solve(self.map, torque)
                except np.linalg.LinAlgError:
                    pass
                self.evaluation_step += 1
                self.step += 1
                return np.clip(torque / self.limits, -1.0, 1.0)
            mujoco.mj_forward(self.model, self.data)
            jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacobian, None, self.site)
            jacobian = jacobian[:, :7]
            position = self.data.site_xpos[self.site]
            velocity = jacobian @ dq
            desired = np.asarray(observation["target"]["position"], dtype=float).copy()
            desired_force = float(observation["target"]["normal_force"])
            measured_force = float(observation["contact_force"])
            if self.scenario == "contact":
                self.contact_offset = np.clip(
                    self.contact_offset + 0.012 * (measured_force - desired_force) * self.dt,
                    0.0, 0.15,
                )
                desired[2] += self.contact_offset
            contact_mode = self.scenario == "contact" and (desired[2] < 0.52 or measured_force > 0.5)
            stiffness = np.array([420.0, 420.0, 80.0 if contact_mode else 420.0])
            damping = np.array([42.0, 42.0, 45.0 if contact_mode else 42.0])
            wrench = stiffness * (desired - position) - damping * velocity

            self.data.qacc[:] = 0.0
            mujoco.mj_inverse(self.model, self.data)
            torque = self.data.qfrc_inverse.copy() + self.bias_correction + jacobian.T @ wrench
            nullspace = np.eye(7) - jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-4 * np.eye(3), jacobian
            )
            torque += nullspace @ (-7.0 * (q - self.home) - 2.0 * dq)
            try:
                torque = np.linalg.solve(self.map, torque)
            except np.linalg.LinAlgError:
                pass
            self.evaluation_step += 1
        self.step += 1
        return np.clip(torque / self.limits, -1.0, 1.0)


def make_controller():
    return GoldenController()
