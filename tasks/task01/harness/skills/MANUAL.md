# Harness library — manual

A library ships in this container. It is ordinary Python in your own process, driving the
simulator through the same metered socket you would have used anyway.

It buys you three things: you do not have to write servo loops, you do not have to turn a
depth map into metres by hand, and you can ask what is in a picture in words. It buys you
**nothing else**. Everything here sees the observation you see, spends the budget you
spend, and cannot tell you anything the simulator did not already show you — the
segmenter is a pure function of the frame you hand it, with no route to the simulator's
state.

There is no `pick("mug")`. Deciding what to manipulate, finding it, and choosing how to
take hold of it is the task — these are the parts you would otherwise have written first.

## The tiers

```python
from harness.client import SimClient, ObsSpec
from harness.skills import camera, geometry, perception, transforms
from harness.skills import reach, move_eef, grasp, release, lift, move_base, settle
```

### `perception` — words to pixels

`segment_by_text(rgb, "coffee mug")` `segment_by_boxes(rgb, boxes)` `plan_grasp(points)`

SAM3 and Contact-GraspNet. **Free in both phases** — they touch no simulator, so nothing
they do is charged against the interaction budget. They cost wall clock, which is also a
budget: tens of milliseconds on a big frame, paid every call.

`segment_by_text` returns `[{"mask", "box", "score"}]`, best first, **empty when nothing
matched** — the common case for a phrasing the model does not know or an object out of
view, and worth checking before you index. The episode instruction names the object; that
phrasing is a good first prompt. When words fail but you can see the thing, box it:
`segment_by_boxes` takes pixel corners, the same format results come back in.

`plan_grasp` takes a point cloud and returns `(K, 4, 4)` poses **in the frame you passed
in**, `+z` the approach direction. Convert to the base frame first if you mean to rank by
`geometry.select_top_down_grasp`, which needs `+z` up to know what "down" means.

### `transforms` — frames

`quat_to_mat` `mat_to_quat` `mat_to_axisangle` `make_transform` `invert_transform`
`decompose_transform` `transform_points` `base_pose_in_world` `world_to_base`
`base_to_world` `normalize`

Quaternions are **xyzw**, the order every `robot0_*_quat` key uses. Positions are metres.

`world_to_base` is the one that decides runs. Actions are in the robot base frame and the
cameras see the world; skipping that rotation raises nothing, and the arm simply drifts.

`mat_to_axisangle(R_des @ R_cur.T)` is the command that turns `R_cur` toward `R_des`
-- the rotation-vector form the action's rotation channels expect.

### `camera` — pixels to metres

`cloud_in_base(obs, camera, mask=...)` `pixel_to_base_point(obs, camera, u, v)`
`base_to_pixel(obs, camera, point)` `camera_pose_in_base(camera, obs)`
`intrinsics(camera, height)` `deproject(depth, K)`

`cloud_in_base(obs, camera, mask=...)` is the whole pipeline in one call: deproject, drop
unusable pixels, place in the frame your actions are in. Give it a mask from `perception`
and it returns just that object.

**The extrinsics are included**, measured off the robot: the agent views are fixed in the
base frame, and the wrist camera is fixed in the hand frame, so it needs the observation
to place. Depth is **z-depth**, along the optical axis rather than along the pixel's own
ray, which is what these functions assume.

Doubting them is cheap: `cloud_in_base` each camera separately and histogram the `z`. All
three see the countertop from different poses, so they must agree on its height.

### `geometry` — clouds to targets

`remove_support_plane(points)` `cluster_points(points)` `oriented_bbox(points)`
`select_top_down_grasp(grasps, scores)`

A kitchen cloud is mostly counter, so removing the dominant horizontal surface before
clustering is usually the difference between one object and one blob. All deterministic:
no RANSAC and no sampling, so the same cloud gives the same answer every run.
`oriented_bbox` is gravity-aligned — objects sit on surfaces, and a free 3-DoF fit on a
cloud that only sees one face tips the box toward it.

### `primitives` — motion

`reach(sim, target, gripper=..., rot=...)` `move_eef(sim, delta, gripper=..., rot=...)`
`grasp(sim)` `release(sim)` `lift(sim, h)`
`move_base(sim, dx, dy, dyaw, hold_eef=..., gripper=...)` `settle(sim, gripper=...)`

They do not look anything up: you supply the target, and in exchange they behave the same
way every time. **All positions are in the robot BASE frame** — the frame your actions are
in, and the frame `robot0_base_to_eef_pos` reports.

**`gripper` is required wherever it appears**: -1 opens, +1 closes, held on every step.
There is no neutral gripper command — it is a target, not a rate — so say what the hand
should be doing; +1 while carrying, or the grasp comes off.

They request `cameras=()`, so they never render, which makes each step markedly cheaper in
wall clock. Look with `sim.observe()` between calls.

Each gives up when it stops making progress rather than spending its whole `max_steps`,
because that allowance is your budget — `max_steps` is a ceiling, not a plan.

## What a primitive returns

`sim.step`'s own reply for the last step it took, plus `outcome`:

| field | meaning |
|---|---|
| `outcome` | `"reached"`, `"gave_up"` or `"blocked"` — the skill's own opinion |
| `steps` | interaction it spent, charged to your budget. `0` if it needed none |
| `success` | **the environment's** predicate, never the skill's. `None` if no step was taken |
| `episode_over` | the episode cannot continue; reset in development, or advance with `next_trial()` in evaluation |
| `ended` | why the episode ended, or `None` while it is alive |
| `pos_err`, `rot_err` | the servo's final errors, metres and radians |
| `gap` | after `grasp`/`release`: the finger opening, metres. An empty hand closes to ~6 mm; more means SOMETHING is between the pads — not proof it is the right thing |
| `eef_drift` | after `move_base(hold_eef=True)`: how far the pinned hand strayed, metres in the world |

A primitive returns the moment the episode ends, so it never steps a dead one. Started on
an already-finished episode, it returns `blocked` without acting.

## Things that will cost you a run if you skip them

- **`settle(sim)` before anything precise.** MuJoCo does not stop when the command does. A
  grasp attempted while the arm is still ringing is the most common way a correct plan
  fails.
- **`grasp` cannot tell you it caught anything.** It reports that the fingers stopped
  closing, which an empty hand does too. Whether the object came with them is a perception
  question — `lift` and look.
- **Base motion disturbs the arm.** `move_base`, then `settle`.
- **`reach` does not move the base.** A target outside the arm's envelope comes back
  `gave_up`; `move_base` first.
- **`reach(..., rot=)` decides how the hand arrives**, not just where: a base-frame 3x3
  matrix or xyzw quaternion, column 2 the approach direction, column 1 the closing line.
- **Driving the base while holding something attached to the world needs
  `hold_eef=True`** — it pins the gripper's world pose while the base moves; plain
  `move_base` drags the hand along and the grasp comes off. Keep each held move short
  (~0.15 m) and check `eef_drift` in the reply.
- **The gains were calibrated on ONE scene, unloaded.** A motion that oscillates or
  crawls is a gain wrong for your situation, not necessarily a fault in your target.
- **`sim.observe()` is free**, in both phases, so look before you act rather than acting to
  see.

## Worked example — and it is a demonstration, not a solution

```python
from harness.client import SimClient, ObsSpec
from harness.skills import camera, geometry, perception
from harness.skills import reach, grasp, lift, settle

CAM = "robot0_agentview_left"

with SimClient() as sim:
    if sim.status()["phase"] == "development":
        sim.reset()
    look = sim.observe(ObsSpec(width=256, depth=True))
    obs = look["obs"]

    hits = perception.segment_by_text(obs[f"{CAM}_image"], "mug")
    points = camera.cloud_in_base(obs, CAM, mask=hits[0]["mask"])   # hits may be empty
    clusters = geometry.cluster_points(geometry.remove_support_plane(points))

    box = geometry.oriented_bbox(clusters[0])
    reach(sim, box["center"] + [0.0, 0.0, 0.10], gripper=-1.0)
    settle(sim, gripper=-1.0)
    reach(sim, box["center"], gripper=-1.0)
    grasp(sim)
    lift(sim, 0.15)
```

None of those lines is arithmetic you have to get right. Every decision around them still
is: which words to use, which of several hits is the right one, whether to approach from
above at all, whether the grasp held. That is the task.

These are functions in a file. Call them, wrap them, copy one and change it, or skip the
library and write your own loop.
