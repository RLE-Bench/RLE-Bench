"""RGB and robot controls for the tabletop experiments.

TabletopClient controls a PandaOmron with an arm, gripper, mobile base, and torso.
Control runs at 20 Hz. The table top is z=0.9 m, spanning 0.7 × 1.6 m around
world (0, 0). Quaternions use unit xyzw. Images are upright 512 × 512 RGB arrays.

step() uses 12 values in [-1, 1]: base-frame arm translation (3, scaled by
0.05 m), axis-angle rotation (3, scaled by 0.5 rad), gripper (+1 closes,
-1 opens), base forward/lateral/yaw velocity (3), torso velocity (1), and
mode (-1 arm, +1 base). Neutral hold: [0]*6 + [-1] + [0]*4 + [-1].
move() instead tracks a world-frame tool pose through the same controller.
Use intermediate waypoints; collisions and reach limits can obstruct motion.
No automatic grasping or perception tools are provided. BalanceCoins also
provides cube_positions and pan_positions in world coordinates.

A client owns exclusive control until disconnect() or its with block exits.
Disconnecting preserves the scene; a second client receives RemoteError(kind="busy").

HiddenCOMClient uses a fixed arm (7-D actions) and submit('A'|'B'|'C'|'D'); no reset.
Observe every new box before acting. SpeedrunClient exposes the cube's 14-D controls;
task_info() describes its tilted robot frames and cameras. Its reset forfeits the trial.
"""
from rlebench.runtime.client import SimClient, ObsSpec, RemoteError


def tabletop_view(result):
    obs = result.pop("obs", {})
    result["images"] = {}
    aliases = {"joint_pos": "joint_pos", "joint_vel": "joint_vel", "eef_pos": "eef_pos",
               "eef_quat": "eef_quat", "gripper_qpos": "gripper_pos", "base_pos": "base_pos",
               "base_quat": "base_quat", "base_to_eef_pos": "eef_base_pos", "base_to_eef_quat": "eef_base_quat"}
    cameras = {"robot0_agentview_left": "left", "robot0_agentview_right": "right",
               "robot0_eye_in_hand": "wrist"}
    for key, value in obs.items():
        if key.endswith("_image"):
            camera = key[:-6]
            result["images"][cameras.get(camera, camera)] = value
        elif key.startswith("robot0_") and key[7:] in aliases:
            result[aliases[key[7:]]] = value
        elif key in ("cube_positions", "pan_positions"):
            result[key] = value
    result["done"] = result.get("phase") == "finished"
    return result


class TabletopClient(SimClient):
    def observe(self):
        """Return images (left, right, wrist), joint_pos/joint_vel, eef_pos/eef_quat,
        gripper_pos, base_pos/base_quat, and eef_base_pos/eef_base_quat. Positions
        are in meters; eef_base_* is relative to the robot base, other poses are
        world-frame. Includes steps_used, steps_remaining, interaction_budget,
        done, ended, and live. Observation advances no physics. An ended attempt
        returns live=False and no images or proprioception.

        BalanceCoins includes cube_positions (cube centers) and pan_positions
        (left/right upper-surface centers), keyed by name, in world meters.
        No depth, force/torque, contact, mass, or task-quality readings are available.
        The common step response's success field is not a tabletop quality verdict;
        finish() submits the final scene for private scoring.
        """
        return tabletop_view(super().observe(ObsSpec(width=512)))

    def step(self, action, steps=1):
        """Repeat an action, or execute a batch of distinct actions, then observe.

        Tabletop actions have 12 components; HiddenCOM actions have seven.
        Batches contain 1–200 steps and cannot exceed the remaining budget.
        For a batch, leave steps=1. HiddenCOM uses the last observed trial ID;
        stale actions are rejected.
        """
        if hasattr(action, "tolist"):
            action = action.tolist()
        if action and isinstance(action[0], list):
            if steps != 1:
                raise ValueError("steps must be 1 for a batch")
        else:
            if type(steps) is not int or not 1 <= steps <= 200:
                raise ValueError("steps must be 1–200")
            action = [action]*steps
        return tabletop_view(super().step(action, ObsSpec(width=512)))

    def move(self, position, quaternion, gripper=1, steps=40):
        """Track a world-frame tool pose for 1–200 metered steps, then observe.

        position is xyz in meters, quaternion is unit xyzw. Gripper +1 closes,
        -1 opens. The mobile base and torso hold still during this movement;
        HiddenCOM uses a fixed arm. Use intermediate waypoints as needed;
        collisions and reach limits can obstruct movement.
        """
        return tabletop_view(self._request("move", position=list(position),
            quaternion=list(quaternion), gripper=gripper, steps=steps))

    def reset(self):
        """Restore the same initial tabletop scene and observe; costs one step."""
        return tabletop_view(super().reset())

    def finish(self):
        """Submit the final simulator state and return status. Release the structure first."""
        return self.finish_trial()


class SpeedrunClient(SimClient):
    """The cube has one attempt. reset forfeits it; recover_drop costs one step."""
    def reset(self, seed=None):
        return self.finish_trial()

    def recover_drop(self):
        return self._request("recover_drop")

    def phase(self):
        return self.status()["phase"]

    def trial_info(self):
        return {**self.task_info(), **self.observe()}


class HiddenCOMClient(TabletopClient):
    """Three 0.15 × 0.15 × 0.05 m boxes, initially at (0, 0, 0.779) m.

    Each box sits on a 4 mm mat above the z=0.75 m table. Its rigid centered
    handle runs along x: an 80 × 16 × 12 mm bar at (0, 0, 0.860) m with 50 mm
    clearance below. You may push, lift, or rotate the box on the clear tabletop.
    The top camera looks straight down from (0.03, 0, 1.71) m with 43° vertical FOV.

    Actions have seven components in [-1, 1]: three world-aligned position deltas
    scaled by 0.05 m, three axis-angle deltas scaled by 0.5 rad, and gripper
    (+1 closes, -1 opens). Control runs at 20 Hz. Movement returns an observation.
    Actions use the last observed trial ID; stale actions are rejected. Observe
    every new box before acting. Observation and submission advance no physics.
    """
    def _request(self, op, **fields):
        if op in ("step", "move", "submit"):
            if not hasattr(self, "_trial"):
                self.observe()
            fields.setdefault("trial", self._trial)
        result = super()._request(op, **fields)
        if "obs" in result:
            self._trial = result["trial_index"]+1
            result.update(trial=self._trial, trial_count=result["total_trials"],
                          remaining_steps=result["steps_remaining"], steps=result["steps_used"], submitted=False)
        return result

    def observe(self):
        """Return three 512 × 512 RGB arrays under images (workspace, closeup, top),
        joint_pos/joint_vel, eef_pos/eef_quat (xyzw), gripper_pos, trial,
        trial_count, steps, remaining_steps, and total_steps, without advancing
        physics. No force/torque, object-pose, mass, or contact readings are available.
        If live=False, images and proprioception are absent.
        """
        result = super().observe()
        self._trial = result["trial_index"]+1
        return {**result, "trial": self._trial, "trial_count": result["total_trials"],
                "remaining_steps": result["steps_remaining"], "submitted": False}

    def submit(self, quadrant, trial=None):
        """Commit A/B/C/D once, without correctness feedback. Observe the next box.

        After sim_error, any valid answer advances without credit for the failed box.
        done=True means all boxes have ended. trial, if supplied, must match the
        current box; otherwise the last observed trial ID is used.
        """
        fields = {} if trial is None else {"trial": trial}
        return self._request("submit", quadrant=quadrant, **fields)

    def reset(self):
        raise ValueError("this task has no reset")

    def finish(self):
        raise ValueError("submit an answer for the current box")
