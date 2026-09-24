"""Fixed-size rendering and a single-threaded robosuite adapter."""
import random
import re
import os

import numpy as np
from PIL import Image

from .compat import _fix_render_cleanup

RESOLUTION = 512
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
ROBOT_FIELDS = frozenset(("joint_pos", "joint_pos_cos", "joint_pos_sin", "joint_vel",
    "joint_acc", "gripper_qpos", "gripper_qvel", "eef_pos", "eef_quat", "eef_quat_site",
    "base_pos", "base_quat", "base_to_eef_pos", "base_to_eef_quat", "proprio-state"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def prepare():
    import robosuite.macros as macros
    macros.IMAGE_CONVENTION = "opencv"
    _fix_render_cleanup()


def make_kitchen(descriptor):
    import rlebench_ro_assets
    rlebench_ro_assets.install()
    from robocasa.utils.env_utils import create_env
    options = dict(env_name=descriptor["task"], split=descriptor["split"],
                   seed=descriptor["seed"], camera_names=list(CAMERAS),
                   camera_heights=RESOLUTION, camera_widths=RESOLUTION,
                   camera_depths=False, renderer="mujoco")
    if descriptor.get("scene"):
        options.update(split=None, obj_instance_split="target",
                       layout_and_style_ids=[descriptor["scene"]])
    return create_env(**options)


def resize(array, width, height, depth=False):
    if array.shape[:2] == (height, width):
        return array.copy()
    filter_ = Image.Resampling.NEAREST if depth else Image.Resampling.BOX
    return np.asarray(Image.fromarray(array).resize((width, height), filter_)).copy()


class Backend:
    cameras = CAMERAS

    def __init__(self, **descriptor):
        self.descriptor = descriptor
        self.seed = descriptor["seed"]
        self.default_resolution = descriptor.get("default_resolution", 512)
        self.sequence = 0
        self.frames = {}
        self.raw = {}
        self.video = None
        if descriptor.get("media_name") and os.environ.get("RLEBENCH_MEDIA", "1") != "0":
            from .runtime.media import Video
            self.video = Video('/var/lib/rlebench/media/'+descriptor["media_name"]+'.mp4')
        prepare()
        seed_all(self.seed)
        self.env = self.create()

    def create(self):
        return make_kitchen(self.descriptor)

    def reset(self, seed=None):
        self.seed = self.seed if seed is None else seed
        seed_all(self.seed)
        self.env.rng = np.random.default_rng(self.seed)
        self.raw = self.env.reset()
        # Camera observables are configured at the ceiling but rendered only on demand.
        for name, observable in getattr(self.env, "_observables", {}).items():
            if name.endswith(("_image", "_depth")):
                observable.set_enabled(False)
        model = self.env.sim.model
        model.vis.global_.offwidth = RESOLUTION
        model.vis.global_.offheight = RESOLUTION
        self.sequence += 1
        self.frames.clear()
        self.after_reset()
        return dict(info=self.info(), evidence=self.evidence())

    def after_reset(self):
        pass

    def info(self):
        meta = getattr(self.env, "get_ep_meta", lambda: {})()
        return dict(instruction=meta.get("lang"), action_dim=len(self.env.action_spec[0]))

    def evidence(self):
        return dict(success=bool(self.env._check_success()))

    def act(self, action=None, move=None, recover=False):
        if move is not None:
            action = self.move_action(move)
        if recover:
            self.env.recover_drop()
        self.raw, _, done, _ = self.env.step(np.asarray(action, dtype=float))
        if not np.isfinite(self.env.sim.data.qpos).all():
            raise RuntimeError("nonfinite simulator state")
        self.sequence += 1
        self.frames.clear()
        return {**self.evidence(), "done": bool(done)}

    def shown(self):
        return {k: v for k, v in self.raw.items()
                if (match := re.fullmatch(r"robot\d+_(.*)", k)) and match[1] in ROBOT_FIELDS}

    def render(self, camera):
        if camera not in self.frames:
            rgb, depth = self.env.sim.render(camera_name=camera, width=RESOLUTION,
                                              height=RESOLUTION, depth=True)
            model = self.env.sim.model
            near = float(model.vis.map.znear * model.stat.extent)
            far = float(model.vis.map.zfar * model.stat.extent)
            metres = near / (1. - np.asarray(depth, dtype=np.float64) * (1. - near / far))
            self.frames[camera] = (np.asarray(rgb, dtype=np.uint8)[::-1].copy(),
                                   metres[::-1].astype(np.float32).copy())
        return self.frames[camera]

    def observe(self, spec):
        w = min(spec.get("width") or spec.get("height") or self.default_resolution, RESOLUTION)
        h = min(spec.get("height") or spec.get("width") or self.default_resolution, RESOLUTION)
        cameras = self.cameras if spec.get("cameras") is None else spec["cameras"]
        out = self.shown()
        intrinsics = {}
        for camera in cameras:
            rgb, depth = self.render(camera)
            out[camera+"_image"] = resize(rgb, w, h)
            if spec.get("depth"):
                out[camera+"_depth"] = resize(depth, w, h, depth=True)
            cid = self.env.sim.model.camera_name2id(camera)
            f = RESOLUTION / (2 * np.tan(np.deg2rad(self.env.sim.model.cam_fovy[cid]) / 2))
            intrinsics[camera] = [[float(f*w/RESOLUTION), 0, (w-1)/2],
                                  [0, float(f*h/RESOLUTION), (h-1)/2], [0, 0, 1]]
        if self.video and cameras:
            self.video.add(self.render(cameras[0])[0])
        return dict(obs=out, resolution=[w, h], max_resolution=RESOLUTION,
                    intrinsics=intrinsics, state_sequence=self.sequence)

    def finish(self):
        return self.evidence()

    def close(self):
        if self.video:
            self.video.close()
        self.env.close()
