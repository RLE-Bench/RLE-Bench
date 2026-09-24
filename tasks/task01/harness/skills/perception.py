"""Open-vocabulary segmentation and grasp proposal, over a second socket.

Two models run in this container -- SAM3 for text-prompted segmentation, Contact-GraspNet
for grasp candidates -- and this module is the whole of how you reach them. They are the
one thing in the library that is not arithmetic: everything else here you could have
written yourself, and these you could not.

    from harness.skills import perception, camera, transforms

    look = sim.observe(ObsSpec(width=512, depth=True))
    hits = perception.segment_by_text(look["obs"]["robot0_agentview_left_image"], "mug")
    if hits:
        k = camera.intrinsics("robot0_agentview_left", 512)
        pts = camera.mask_to_camera_points(look["obs"]["robot0_agentview_left_depth"],
                                           k, hits[0]["mask"])

THEY ARE FREE, in both phases: they never touch the simulator, so nothing here is charged
against the interaction budget. They cost WALL CLOCK, which is also a budget -- a call is
tens of milliseconds on a big frame, paid every time.

THEY SEE ONLY WHAT YOU SEND. A segmenter is a pure function of the picture you hand it;
it has no route to the simulator's state, and neither do you through it. Naming an object
that is not in frame returns nothing, not a pose.
"""

from __future__ import annotations

import os
import socket
from typing import Any, Sequence

import numpy as np

from .. import perception_protocol as P

DEFAULT_SOCKET = os.environ.get(
    "RLEBENCH_PERCEPTION_SOCKET", "/run/rlebench/perception.sock"
)

__all__ = ["PerceptionError", "segment_by_text", "segment_by_boxes",
           "plan_grasp", "info"]


class PerceptionError(RuntimeError):
    """The perception service refused a request, or is not running."""


def _request(op: str, **fields: Any) -> dict:
    """One request, one reply. A fresh connection each call: these are occasional, and a
    long-lived socket would be one more thing for your code to own."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(DEFAULT_SOCKET)
    except OSError as exc:
        raise PerceptionError(
            f"no perception service at {DEFAULT_SOCKET}: {exc}") from exc
    try:
        P.send(sock, {"op": op, **fields})
        msg = P.LineReader(sock).read()
    finally:
        sock.close()
    if msg is None:
        raise PerceptionError("perception service closed the connection")
    if not msg.get("ok", False):
        raise PerceptionError(str(msg.get("error", "unknown error")))
    return msg


def info() -> dict:
    """What is loaded and on what device. Free, and cheap enough to call at startup."""
    return {k: v for k, v in _request("info").items() if k != "ok"}


def segment_by_text(rgb: np.ndarray, text: str,
                    threshold: float = 0.5) -> list[dict]:
    """Segment everything matching `text` in an RGB frame.

    Args:
        rgb: (H, W, 3) uint8, straight out of the observation.
        text: what to look for -- a noun phrase, e.g. "coffee mug", "cabinet handle".
            The episode instruction names the object; that phrasing is a good start.
        threshold: confidence below which a detection is dropped.

    Returns:
        A list of `{"mask": (H, W) bool, "box": [x1, y1, x2, y2], "score": float}`,
        highest score first, in the frame's own pixel coordinates. **Empty when nothing
        matched** -- which is a real answer and the common one for a phrasing the model
        does not recognise or an object out of view. Check it before indexing.
    """
    out = _request("segment_text", rgb=np.asarray(rgb), text=str(text),
                   threshold=float(threshold))
    return list(out.get("results", []))


def segment_by_boxes(rgb: np.ndarray, boxes: Sequence[Sequence[float]],
                     labels: Sequence[bool] | None = None) -> list[dict]:
    """Segment from boxes you drew instead of words.

    Args:
        rgb: (H, W, 3) uint8.
        boxes: `[[x1, y1, x2, y2], ...]` in PIXELS, the same format results come back in
            -- so a box from one call can be refined in the next.
        labels: True for "this region is the thing", False for "this region is not",
            one per box. All True by default.

    Returns:
        The same shape `segment_by_text` returns.

    Worth having when a name fails but you can see the thing: box a cluster you already
    found, or a region off a saved frame, and ask for its extent. This is the model's
    only non-text prompt -- it takes boxes, not points.
    """
    b = [[float(x) for x in box] for box in boxes]
    lab = [True] * len(b) if labels is None else [bool(x) for x in labels]
    return list(_request("segment_boxes", rgb=np.asarray(rgb), boxes=b,
                         labels=lab).get("results", []))


def plan_grasp(points: np.ndarray, segments: dict | None = None,
               max_grasps: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Propose gripper poses for a point cloud.

    Args:
        points: (N, 3) the scene, in ANY frame you like -- see below.
        segments: optional `{name: (M, 3)}` of the objects you care about. Given these,
            candidates are proposed per object and returned for the whole set; without
            them the whole cloud is treated as one scene, which is slower and less
            focused.
        max_grasps: cap on how many come back, best score first.

    Returns:
        `(grasps, scores)` -- `(K, 4, 4)` homogeneous poses and `(K,)` confidences,
        **in the same frame as the cloud you passed in**. `+z` of each pose is the
        approach direction; `geometry.select_top_down_grasp` ranks them, and expects a
        frame whose `+z` is up, so convert the cloud to the base frame BEFORE calling
        this if that is what you want to rank in.

    `(0, 4, 4)` back means no candidate was found: too few points, or nothing graspable.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    payload: dict[str, Any] = {"points": pts, "max_grasps": int(max_grasps)}
    if segments:
        payload["segments"] = {str(k): np.asarray(v, dtype=np.float32).reshape(-1, 3)
                               for k, v in segments.items()}
    out = _request("plan_grasp", **payload)
    grasps = np.asarray(out.get("grasps", np.zeros((0, 4, 4))), dtype=float)
    scores = np.asarray(out.get("scores", np.zeros((0,))), dtype=float)
    return grasps.reshape(-1, 4, 4), scores.reshape(-1)
