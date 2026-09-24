"""The harness library: tools, not answers.

    perception.py   SAM3 text-prompted segmentation, Contact-GraspNet grasp proposals
    camera.py       pixels + depth -> points in the frame your actions are in
    geometry.py     point cloud -> objects: drop the counter, cluster, box, grasp choice
    transforms.py   frames and rotations -- world_to_base above all
    primitives.py   motion -- reach, move_eef, grasp, release, lift, move_base, settle

There is no `pick("mug")`: deciding what matters, finding it and choosing how to take hold
of it is the task. And there is no second execution path -- a primitive's steps are metered
exactly as yours are, and everything here sees exactly the observation you see.

    from harness.client import SimClient, ObsSpec
    from harness.skills import camera, geometry, perception, transforms
    from harness.skills import reach, grasp, lift, settle

    with SimClient() as sim:
        sim.reset()
        obs = sim.observe(ObsSpec(width=512, depth=True))["obs"]
        hits = perception.segment_by_text(obs["robot0_agentview_left_image"], "mug")
        pts = camera.cloud_in_base(obs, "robot0_agentview_left", mask=hits[0]["mask"])
        reach(sim, geometry.oriented_bbox(pts)["center"], gripper=-1.0)

MANUAL.md, next to this file, is the part worth reading first.
"""

from __future__ import annotations

from . import camera, geometry, perception, transforms
from .primitives import (
    BLOCKED, GAVE_UP, REACHED, grasp, lift, move_base, move_eef, reach, release, settle,
)

__all__ = [
    "camera", "geometry", "perception", "transforms",
    "reach", "move_eef", "grasp", "release", "lift", "move_base", "settle",
    "REACHED", "GAVE_UP", "BLOCKED",
]
