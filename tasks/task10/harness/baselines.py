"""Verifier-side calibration baselines."""

from __future__ import annotations

import mujoco
import numpy as np


class NominalController:
    def reset(self, context, seed):
        self.model = mujoco.MjModel.from_binary_path(context["model_file"])
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site(context["ee_site"]).id
        self.limits = np.asarray(context["torque_limit"])

    def act(self, observation):
        q, dq = np.asarray(observation["qpos"]), np.asarray(observation["qvel"])
        self.data.qpos[:], self.data.qvel[:] = q, dq
        mujoco.mj_forward(self.model, self.data)
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacobian, None, self.site)
        jacobian = jacobian[:, :7]
        error = np.asarray(observation["target"]["position"]) - self.data.site_xpos[self.site]
        self.data.qacc[:] = 0.0
        mujoco.mj_inverse(self.model, self.data)
        torque = self.data.qfrc_inverse.copy() + jacobian.T @ (300.0 * error - 30.0 * (jacobian @ dq))
        return np.clip(torque / self.limits, -1.0, 1.0)


class ZeroController:
    def reset(self, context, seed):
        pass

    def act(self, observation):
        return np.zeros(7)
