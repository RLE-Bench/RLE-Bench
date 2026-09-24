"""Pixels to metres, and into the frame your actions are in.

`cloud_in_base(obs, camera, mask=...)` is the one you want: it deprojects, drops unusable
pixels and applies the extrinsic in one call.

Each camera is specified by its VERTICAL field of view, so `f` follows from `fovy` and the
height you rendered at -- and scales with it, which is why nothing here caches a matrix.

Depth is z-DEPTH: distance to the camera PLANE along the optical axis, not range along the
pixel's own ray.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CAMERAS", "FOVY", "WRIST", "intrinsics", "deproject", "camera_pose_in_base",
           "to_base", "cloud_in_base", "pixel_to_base_point", "base_to_pixel"]

CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)

# Vertical field of view, degrees. The simulator's own numbers.
FOVY = {
    "robot0_agentview_left": 60.0,
    "robot0_agentview_right": 60.0,
    "robot0_eye_in_hand": 75.0,
}

WRIST = "robot0_eye_in_hand"


def _base_name(camera: str) -> str:
    """`robot0_eye_in_hand_image` and `..._depth` both name the same camera."""
    for suffix in ("_image", "_depth"):
        if camera.endswith(suffix):
            return camera[: -len(suffix)]
    return camera


def intrinsics(camera: str, height: int, width: int | None = None) -> np.ndarray:
    """The (3, 3) pinhole matrix for `camera` at the size you rendered.

    `width` defaults to `height`, since `ObsSpec(width=N)` renders a square. Pixels are
    scaled from the fixed square render; rectangular deliveries scale each axis.
    """
    name = _base_name(camera)
    if name not in FOVY:
        raise KeyError(f"unknown camera {camera!r}; expected one of {CAMERAS}")
    h = int(height)
    w = int(height if width is None else width)
    f = (h / 2.0) / np.tan(np.radians(FOVY[name]) / 2.0)
    return np.array([[f * w / h, 0.0, (w - 1) / 2.0],
                     [0.0, f, (h - 1) / 2.0],
                     [0.0, 0.0, 1.0]])


def deproject(depth: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """z-depth map + intrinsics -> an (H, W, 3) grid of points in the camera frame.

    +x right, +y down, +z along the optical axis -- the usual pinhole frame. Pixels with
    non-finite or non-positive depth come back as NaN rather than as a point at the
    origin, so they can be dropped instead of quietly joining a cluster.
    """
    d = np.asarray(depth, dtype=float)
    if d.ndim == 3 and d.shape[2] == 1:
        d = d[:, :, 0]
    k = np.asarray(mat, dtype=float)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    h, w = d.shape
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    valid = np.isfinite(d) & (d > 0)
    z = np.where(valid, d, np.nan)
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)

# --------------------------------------------------------------------- extrinsics
#
# Read off the model, in the computer-vision frame used above (MuJoCo's cameras look down
# -z with +y up, so these carry a half turn about x relative to it).
#
# The agent-view cameras hang off the mobile base, which the torso moves under rather than
# with, so their pose in the base frame is a constant.
# p_base = R @ p_cam + t.
_AGENTVIEW_POSE = {
    "robot0_agentview_left": (
        np.array([[-0.2019736289, -0.5281292977, 0.8247945794],
                  [-0.9792973845, 0.0972611583, -0.1775299971],
                  [0.0135383166, -0.8435755521, -0.5368398289]]),
        np.array([-0.5, 0.35, 1.05]),
    ),
    "robot0_agentview_right": (
        np.array([[0.2019735326, -0.5281292832, 0.8247946122],
                  [-0.9792974037, -0.0972610594, 0.1775299454],
                  [-0.013538365, -0.8435755725, -0.5368397956]]),
        np.array([-0.5, -0.35, 1.05]),
    ),
}

# p_hand = R @ p_cam + t, for the wrist. The hand frame is the one the observation reports:
# origin `robot0_base_to_eef_pos`, orientation `robot0_base_to_eef_quat`. The -0.097 is the
# gap between the hand BODY the camera hangs off and the eef SITE the observation reports.
_HAND_TO_CAM = (
    np.array([[0.0, -1.0, 0.0],
              [1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0]]),
    np.array([0.05, 0.0, -0.097]),
)


def camera_pose_in_base(camera: str, obs: dict | None = None) -> tuple:
    """`(R, t)` taking camera coordinates to base coordinates: `p_base = R @ p_cam + t`.

    The agent-view cameras need no observation -- they do not move. The wrist camera
    does, and raises without one rather than silently returning a stale pose.
    """
    name = _base_name(camera)
    if name in _AGENTVIEW_POSE:
        return _AGENTVIEW_POSE[name]
    if name == WRIST:
        if obs is None:
            raise ValueError("the wrist camera moves with the arm: pass the observation")
        from .transforms import quat_to_mat

        rot_hand = quat_to_mat(obs["robot0_base_to_eef_quat"])
        pos_hand = np.asarray(obs["robot0_base_to_eef_pos"], dtype=float)
        rot_cam, off_cam = _HAND_TO_CAM
        return rot_hand @ rot_cam, rot_hand @ off_cam + pos_hand
    raise KeyError(f"no extrinsics for {camera!r}")


def to_base(points: np.ndarray, camera: str, obs: dict | None = None) -> np.ndarray:
    """Camera-frame points -> base-frame points, which is the frame actions are in."""
    rot, off = camera_pose_in_base(camera, obs)
    p = np.atleast_2d(np.asarray(points, dtype=float))
    return p @ rot.T + off


def cloud_in_base(obs: dict, camera: str, mask: np.ndarray | None = None,
                  stride: int = 2, max_range: float = 4.0) -> np.ndarray:
    """One camera's depth map -> an (N, 3) base-frame point cloud.

    Deproject, drop unusable pixels, place in the base frame. `mask` restricts it to a
    segmentation, `stride` subsamples (a 512x512 frame is 262k points, and clustering that
    is slow), `max_range` drops far background.

    Requires `ObsSpec(depth=True)` -- the depth keys are absent otherwise.
    """
    depth = np.asarray(obs[f"{_base_name(camera)}_depth"], dtype=float)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    mat = intrinsics(camera, depth.shape[0], depth.shape[1])
    points = deproject(depth, mat)[::stride, ::stride]
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        if m.ndim == 3:
            m = m[:, :, 0]
        points = np.where(m[::stride, ::stride, None], points, np.nan)
    flat = points.reshape(-1, 3)
    flat = flat[np.isfinite(flat).all(axis=1)]
    flat = flat[flat[:, 2] < float(max_range)]
    return to_base(flat, camera, obs).reshape(-1, 3)


def pixel_to_base_point(obs: dict, camera: str, u: int, v: int,
                        patch: int = 2) -> np.ndarray | None:
    """One pixel -> a base-frame point, or None where that pixel has no usable depth.

    The depth is the median over a `(2*patch+1)` square rather than the single sample:
    on an object's silhouette one pixel can land on the background metres behind it, and
    a median is what stops a target from jumping there.
    """
    depth = np.asarray(obs[f"{_base_name(camera)}_depth"], dtype=float)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    h, w = depth.shape[:2]
    u, v = int(round(u)), int(round(v))
    window = depth[max(0, v - patch):min(h, v + patch + 1),
                   max(0, u - patch):min(w, u + patch + 1)]
    window = window[np.isfinite(window) & (window > 1e-4)]
    if window.size == 0:
        return None
    z = float(np.median(window))
    mat = intrinsics(camera, h, w)
    cam = np.array([(u - mat[0, 2]) * z / mat[0, 0],
                    (v - mat[1, 2]) * z / mat[1, 1], z])
    return to_base(cam, camera, obs).reshape(3)


def _project(points: np.ndarray, camera: str, height: int,
             width: int | None = None) -> tuple:
    """Camera-frame points -> `(u, v)` pixel arrays."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    mat = intrinsics(camera, height, width)
    z = np.where(np.abs(p[:, 2]) < 1e-6, 1e-6, p[:, 2])
    return (mat[0, 2] + mat[0, 0] * p[:, 0] / z,
            mat[1, 2] + mat[1, 1] * p[:, 1] / z)


def base_to_pixel(obs: dict, camera: str, point_base) -> tuple:
    """Base-frame point -> `(u, v, z)` in one camera. `z` is its depth in that camera.

    The inverse of `pixel_to_base_point`, for checking whether a target you computed is
    where you think it is: project it, then look at that pixel. A NEGATIVE `z` means the
    point is behind the camera, and the pixel is meaningless.
    """
    rot, off = camera_pose_in_base(camera, obs)
    cam = rot.T @ (np.asarray(point_base, dtype=float).reshape(3) - off)
    name = _base_name(camera)
    frame = obs.get(f"{name}_image", obs.get(f"{name}_depth"))
    shape = np.asarray(frame).shape
    u, v = _project(cam[None, :], camera, shape[0], shape[1])
    return float(u[0]), float(v[0]), float(cam[2])
