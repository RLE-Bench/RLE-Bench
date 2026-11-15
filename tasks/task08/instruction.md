# Task: One Mobile Base for Panda, UR5e, and xArm7

Design and build one mecanum mobile-manipulator base compatible with the three
supplied canonical arms: Franka Panda, Universal Robots UR5e, and UFACTORY
xArm7. The same submitted chassis, battery, wheels, and `arm_mount_site` must be
used unchanged with every arm. A universal 180 x 180 x 10 mm adapter plate is
supplied; do not redesign it or count it as submitted chassis geometry.

The verifier removes the submitted preview arm and mounts each trusted arm
with the supplied adapter. Submit `controller.py` to control shelf entry and
loaded target holding. Each episode starts at the fixed initial pose below.
Your controller may reposition both the base and arm before entering. The
verifier independently
measures the resulting motion; other base-design and dynamic tests use its own
controllers.

Each arm is evaluated at all 12 public shelf targets, and every target counts
toward that arm's score. Each trial must carry the required 1 kg payload without
shelf contact, tipping, or sustained loss of wheel support. All trials are
repeated at 2 kg for payload-margin credit. Checkpoints are aggregated by the
minimum across the three arm variants. `shelf_spec.json` defines the exact target
coordinates and names. Its approach points are optional suggestions; your
controller chooses the actual entry path. All coordinates stay in the fixed
world frame when the robot is initialized farther outside.

## Mechanical interface

Create a body named `base` with a free joint named exactly `base_free`. Create a
site named `arm_mount_site` as a direct child of `base`. It marks the underside
center of the supplied adapter; its +z axis is the mounting normal. The verifier
inserts the 0.875 kg plate above this site. Panda and xArm7 use zero adapter yaw;
the provided UR5e hole pattern applies a fixed 180-degree yaw so all arms share
the base's +x forward direction. Each arm's root origin sits on the plate's
upper surface, 10 mm above `arm_mount_site`, with no standalone asset offset.

Your public `robot.xml` must still contain the supplied Panda subtree with
`arm_*` names so it loads in `/workspace`. That subtree is preview-only and is
replaced by trusted arm assets during scoring. Do not include the adapter in
`robot.xml`, because the verifier supplies it automatically.

## Wheels

Use the supplied mecanum wheel components unchanged except for each complete
wheel body's mounting position (`pos`) on the chassis and the required naming
substitution below. Preserve the wheel orientation, hub, all 16 independent
passive rollers per wheel, joints, geometry, masses, contact parameters, and
actuator settings. Do not replace the rollers with a solid cylinder or otherwise
simplify or redesign the wheel structure.

Use the left component for FL and RR, and the right component for FR and RL.
For each wheel, replace `EXAMPLE` in the supplied component names with its corner
`W` (`FL`, `FR`, `RL`, or `RR`). The resulting names must be exactly:

| Element | Required name |
|---|---|
| Wheel body | `wheel_W` |
| Drive joint | `drive_W` |
| Hub geom | `hub_W` |
| Drive actuator | `motor_W` |
| Roller body | `roller_W_i` |
| Roller joint | `rollerj_W_i` |
| Roller geom | `rollerg_W_i` |

Here `i` runs from `0` through `15`, without leading zeros: for example,
`rollerg_FL_0`, not `rollerg_FL_00`. These must be the final compiled MuJoCo
names, including when using XML replication or attachment.

## Design efficiency and resource targets

Design-efficiency scoring favors a compact, lightweight chassis built from the
supplied 2020, 2040, or 4040 aluminum-profile cross-sections. The preferred
targets are at most 0.56 m along each footprint axis, 60 kg assembled mass
without pickup payload for each arm variant, and 12 m of recognized profile
stock. These are soft scoring targets: exceeding them or using unrecognized
chassis geometry reduces design credit rather than invalidating the submission.
Lower chassis mass, smaller footprint area, and shorter recognized profile
length receive more credit.

The chassis must be buildable and stable within the whole workspace of the robot arm.
One chassis member must intersect the center of each `drive_FL`, `drive_FR`,
`drive_RL`, and `drive_RR` joint. A member passing only near the outside of a
wheel does not satisfy this structural requirement. The supplied 10 kg battery
must be rigidly mounted under `base` with the specified dimensions and names.

## Workspace

```text
/workspace/
├── scene.xml
├── shelf_spec.json
└── assets/
    ├── franka_emika_panda/          canonical Panda model and meshes
    ├── universal_robots_ur5e/       canonical UR5e model and meshes
    ├── ufactory_xarm7/              canonical xArm7 model and meshes
    └── mobile_manipulator_components/
        ├── arm_adapter.xml
        ├── mecanum_wheel_left.xml
        ├── mecanum_wheel_right.xml
        ├── battery.xml
        ├── payload.xml
        ├── aluminum_profiles.xml
        └── component_catalog.json
```

The component XML files describe individual components, not an assembled base.
Create `/workspace/robot.xml`; until it exists, `scene.xml` cannot load. The
completed scene must compile in MuJoCo 3.5.0.

## Initial state and targets

Each target is an independent episode. At time zero, the assembled robot has:

- `base_free` world position **(-0.60, 0.00, 0.08) m** and quaternion
  **(1, 0, 0, 0)** in MuJoCo wxyz order (+x forward).
- Zero joint velocities. The arm starts at the angles below, with its position
  actuators commanded to those same angles; wheel controls start at zero.
- The specified payload attached. The shelf face is at world x = 0.40 m.

| Arm | Initial joint angles (radians, canonical joint order) |
|---|---|
| Panda | `[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]` |
| UR5e | `[-0.83943, -1.22370, -1.33717, -1.07738, -3.03934, 3.03382]` |
| xArm7 | `[-2.26731, -0.62319, -0.00416, 0.22692, 0.02608, 2.72839, -0.08641]` |

Write the approach and hold logic for the 12 targets listed in
`shelf_spec.json`. The verifier resets to this same initial state for each
arm, target and payload, then directly executes your controller from time zero.
It does not reposition or settle the robot before your first action. Your
controller may settle, unfold, drive and approach as needed within the 30 s
budget. Read the supplied initial state and target from `context`.

## Shelf controller

Submit one self-contained `controller.py` using Python, NumPy, SciPy, or MuJoCo.
It must export `make_controller()`, returning an object with:

```python
class Controller:
    def reset(self, context: dict, seed: int) -> None:
        ...

    def act(self, observation: dict):
        return controls  # one finite number per actuator, in actuator_names order
```

`reset` runs once in a fresh process for each target and payload. `context`
contains:

- `model_file`: a read-only MuJoCo MJB copy of the assembled robot and public
  shelf. Load it with `mujoco.MjModel.from_binary_path`. Changes to this local
  model or its data cannot affect the evaluator's simulation.
- `arm`: `panda`, `ur5e`, or `xarm7`; `target` (world xyz), `target_name`,
  `shelf_face_x`, and `payload_kg` (1 or 2).
- `actuator_names`, `arm_joint_names`, `arm_actuator_names`,
  `wheel_actuator_names` (FL, FR, RL, RR), `base_joint`, and `ee_site`.
- Initial `qpos`, `qvel`, and `ctrl`; `control_dt` (0.02 s), `duration` (30 s),
  and `seed` (0). The actual outside starting position is in `qpos`.

`act` receives `time` (elapsed episode seconds), `qpos`, `qvel`, and `ctrl` as
JSON-compatible numbers/lists. Return the full actuator control vector at 50 Hz.
Arm position actuators take joint angles in radians; wheel velocity actuators
take rad/s. Inspect the model for exact actuator types, ordering and limits.
The verifier clamps limited controls and enforces the trusted force limits,
then advances physics at a 0.002 s timestep. You cannot supply generalized
forces, change the state, move the shelf, or report your own success.

Starting outside, reach the target and hold the flange within 0.06 m for one
continuous second with positive stability margin. The payload is attached
throughout. Shelf contact above 2 N, tipping, or sustained loss of support fails
the episode, including during approach. A controller exception, malformed or
non-finite output, or timeout also fails that episode. Missing controllers earn
zero shelf-integration credit; other design checkpoints are still evaluated.

Each process has 60 CPU seconds in total, up to 30 wall seconds for startup and
`reset`, and up to 2 wall seconds per `act` response. It has a 2 GiB address-space
limit, no network, no subprocesses, and no access to verifier files. These are
execution limits, not speed rewards. Use the supplied seed for every random
choice; do not depend on wall-clock time or persistent state. The first target
is replayed for each arm; inconsistent action streams invalidate that arm's
shelf-integration credit. Only `controller.py` is loaded as submitted Python;
keep all controller helpers in that file (maximum 1 MiB).

## Submission and time budget

Agent execution has a hard timeout of **7200 seconds (2 hours)**. Only files
under `/logs/artifacts/` are collected. Copy a loadable `robot.xml` there early,
then update it as the design improves.

```text
/logs/artifacts/robot.xml
/logs/artifacts/controller.py
```

Keep `meshdir="assets/franka_emika_panda/assets"`; the verifier overlays all
canonical assets and ignores submitted asset replacements.
