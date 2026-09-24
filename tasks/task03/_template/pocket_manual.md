# Two-arm cube simulator

Use `harness.client.SpeedrunClient` and `harness.obs.ObsSpec`.
`sim.task_info()` publishes the action layout, the camera calibration, the
robot base poses, and the budgets. Start with `observe()`; the single trial is
already active, and `reset()` ends it.

Each action has 14 values in [-1, 1]: robot0 translation (3), rotation (3),
gripper (1), followed by the same seven values for robot1. Translation and
axis-angle rotation deltas are expressed in each robot's base frame. Full-scale
deltas are 0.05 m and 0.5 rad per control step. Positive gripper closes;
negative opens. Control runs at 20 Hz.

The only camera is `front`.
Images are upright RGB, 512×512 by default. Use `ObsSpec(cameras=())` for
proprioception only, or `ObsSpec(depth=True)` for aligned depth in metres.
Calibration uses OpenGL camera axes: x right, y up, viewing along negative z.
Images use x right and y down. Intrinsics scale with the requested resolution.

Both robots' joint positions and velocities, end-effector poses, and gripper
positions are available. End-effector poses are in world coordinates and
quaternions are xyzw. The simulator does not disclose cube poses, cubie
orientations, or face labels.

```python
import numpy as np
from harness.client import SpeedrunClient
from harness.obs import ObsSpec

with SpeedrunClient() as sim:
    print(sim.task_info())
    observation = sim.observe(ObsSpec(cameras=("front",)))["obs"]
    result = sim.step(np.zeros(14).tolist())
```
