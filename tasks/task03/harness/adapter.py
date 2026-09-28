"""Tabletop scenes supply physics and evidence to the common runtime."""
import numpy as np
from scipy.spatial.transform import Rotation

from .backend import Backend, CAMERAS


class Adapter(Backend):
    def create(self):
        mode = self.descriptor["mode"]
        if mode == "pocket":
            from .tabletop.pocket.recovery import RecoveryPocketCube
            self.cameras = ("front",)
            return RecoveryPocketCube(seed=self.seed)
        if mode == "hidden_com":
            from .tabletop.hidden_com.config import CASES
            from .tabletop.hidden_com.scene import HiddenCOM, CAMERAS as hidden_cameras
            self.cameras = tuple(hidden_cameras)
            self.quadrant = str(np.random.default_rng(self.seed).choice(list("ABCD")))
            return HiddenCOM(self.quadrant, case=CASES[self.descriptor["case"]])
        from .tabletop import make
        return make(self.descriptor["task"], seed=self.seed, camera_names=CAMERAS)

    def info(self):
        info = super().info()
        if self.descriptor["mode"] == "pocket":
            from .tabletop.pocket.front import BASE_POSITIONS, BASE_ROTATIONS
            info.update(robot_base_poses=[dict(pos=list(p), quat_xyzw=r.as_quat().tolist())
                        for p, r in zip(BASE_POSITIONS, BASE_ROTATIONS)],
                        action_reference_frame="each robot's fixed tilted base")
            model, data = self.env.sim.model, self.env.sim.data
            calibration = {}
            for camera in self.cameras:
                cid = model.camera_name2id(camera)
                f = 512/(2*np.tan(np.deg2rad(model.cam_fovy[cid])/2))
                calibration[camera] = dict(width=512, height=512,
                    intrinsics=[[float(f),0,255.5],[0,float(f),255.5],[0,0,1]],
                    position_world=data.cam_xpos[cid].tolist(),
                    rotation_world_from_opengl_camera=data.cam_xmat[cid].reshape(3,3).tolist())
            info["camera_calibration"] = calibration
        return info

    def evidence(self):
        mode = self.descriptor["mode"]
        if mode == "hidden_com":
            return dict(quadrant=self.quadrant, success=False)
        if mode == "tabletop":
            return dict(success=False)  # Quality hooks can move the scene; run only on finish.
        return dict(success=bool(self.env._check_success()), counters=self.env.private_counters())

    def finish(self):
        if self.descriptor["mode"] == "tabletop":
            quality = float(self.env._trial_score())
            if not np.isfinite(quality):
                raise RuntimeError("nonfinite quality")
            return dict(quality=max(0., min(1., quality)))
        return self.evidence()

    def recovery_available(self):
        try:
            self.env.require_dropped()
        except ValueError:
            return False
        return True

    def shown(self):
        public = super().shown()
        if self.descriptor["task"] == "BalanceCoins":
            public.update(self.env.position_observation())
        if self.descriptor["mode"] == "hidden_com":
            robot = self.env.robots[0]
            data = self.env.sim.data
            site = robot.eef_site_id["right"]
            public.update(robot0_eef_pos=data.site_xpos[site].copy(),
                robot0_eef_quat=Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)).as_quat(),
                robot0_gripper_qpos=data.qpos[robot._ref_gripper_joint_pos_indexes["right"]].copy())
        return public

    def move_action(self, move):
        orientation = Rotation.from_quat(move["quaternion"]).as_matrix()
        if self.descriptor["mode"] == "hidden_com":
            data = self.env.sim.data
            site = self.env.robots[0].eef_site_id["right"]
            dp = (move["position"]-data.site_xpos[site]) / .05
            dr = Rotation.from_matrix(orientation @ data.site_xmat[site].reshape(3, 3).T).as_rotvec() / .5
            return np.r_[np.clip(dp, -1, 1), np.clip(dr, -1, 1), move["gripper"]]
        base = Rotation.from_quat(self.raw["robot0_base_quat"]).as_matrix()
        target = base.T @ (move["position"]-self.raw["robot0_base_pos"])
        current = Rotation.from_quat(self.raw["robot0_base_to_eef_quat"]).as_matrix()
        dp = (target-self.raw["robot0_base_to_eef_pos"]) / .05
        dr = Rotation.from_matrix(base.T @ orientation @ current.T).as_rotvec() / .5
        return np.r_[np.clip(dp, -1, 1), np.clip(dr, -1, 1), move["gripper"], 0, 0, 0, 0, -1]
