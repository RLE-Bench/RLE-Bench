# Rubik's Cube — one continuous solve

Write a controller that solves a scrambled 64 mm 2×2×2 Rubik's Cube using two
fixed Panda arms. The cube begins on a support at the center of the table.
The cube has eight corner pieces and 24 stickers, with no centers, edges, or grasping shafts.
Grasp opposing halves and rotate one half relative to the other.
Turn layers through gripper contact and reorient the cube as needed to inspect
hidden faces. Finish with all six faces solved and aligned, and the cube held
still above the table. It may be held by a gripper or rest on the support.

One legal scrambled cube is already active when you connect.
There is no direct cube actuator, face-turn command, or disclosed cube state.
Your code and working files belong in `/workspace` and remain available throughout this attempt.

## Simulator

The simulator runs in a separate privileged process and is available only
through the metered client. Read `/opt/HARNESS_MANUAL.md` for the control and
camera conventions.

```python
from harness.client import SpeedrunClient
from harness.obs import ObsSpec

with SpeedrunClient() as sim:
    info = sim.task_info()
    obs = sim.observe(ObsSpec(cameras=("front",)))["obs"]
    print(sim.status())
```

Actions contain 14 values: robot0's six OSC pose deltas and gripper command,
then robot1's six pose deltas and gripper command. Actions use each arm's base
frame. Both bases tilt outward by 45 degrees. Read robot_base_poses from task_info;
transform world-frame deltas into each tilted base frame.
Control runs at 20 Hz. All values lie in [-1, 1].

The only camera observation is `front_image`,
upright RGB at 512×512, plus both arms' joint, end-effector, and gripper states.
Camera calibration is published by `sim.task_info()`. Use `ObsSpec(depth=True)`
for aligned depth in metres, or `ObsSpec(cameras=())` for proprioception only.
Free observations advance no physics. Each `sim.step(action)` advances one
control step; batching actions charges every step.

## Single-episode contract

Solve this one active cube within 50,000 simulator steps. There is no development
phase, no subsequent battery, and no second episode. The API uses its internal
"evaluation" phase solely to enforce this one-trial lifecycle; it does not mean
you should wait for another stage.

Start with observe(), not reset(). Do not call reset() or end_development().
Reset abandons this only attempt and cannot provide another cube. You may
write, debug and improve your controller online, and recover the same cube
after slip or dropping it. All physical interaction shares the 50,000-step cap.
Free observations do not advance physics. The wall-clock safety limit is nine
hours; inspect status() for the remaining budget.

Use only the front RGB camera, optional aligned depth, and public robot state.
Physically manipulate the cube to reveal hidden faces. Do not access private
simulator state or invoke ideal face turns. Keep code and debug outputs in
/workspace. Aim for a complete physical solve and stop once the simulator
reports success. Symbolic completion or visual similarity alone is insufficient.
The final reward is binary simulator-confirmed success for this single episode.

## Dropped-cube rescue

If the cube falls below 0.90 m, call the no-argument rescue helper:

    from cube_recovery import recover_drop
    result = recover_drop(sim)

This restores the last legally aligned cube configuration recorded before the
drop, upright on the support, and returns both arms to their initial poses.
Completed face turns are retained; an unfinished turn may be rolled back.
The global cube orientation may change: observe the front camera again, and
reinitialize your controller's pose/gripper targets before continuing.
The helper costs one of the SAME 50,000 steps. It does not restart the episode,
reroll the scramble, replenish any budget, or expose cube state. Calling it
while the cube is not dropped or after the episode ends is refused.
Use this helper instead of spending thousands of actions on table pickup.
The ordinary reset() still ends the only trial.

## Minimize rotations

First achieve a physical solve; among successful approaches, use as few face
quarter turns as possible (90/270 degrees = 1, 180 degrees = 2).
Extra face turns reduce your score even when the cube is solved.
Reconstruct the state from camera observations, plan a shortest or demonstrably
short solution before execution, and replan when the observed state changes.
Also minimize whole-cube observation rotations and unnecessary handoffs.
Keep separate records of planned face turns, executed face turns, observation
reorientations, and recoveries. Do not claim optimality without proving it.
The verifier privately records recognizable completed face transitions; it
cannot infer exact motion counts for arbitrary malformed/intermediate states.
Solving the cube is required for a nonzero score; the score also depends on
the number of completed face turns.

## Efficient, feedback-driven motion

The OSC pose interface runs at 20 Hz. A normalized translational action of
+/-1 requests a +/-0.05 m target increment on that base-frame axis; a rotational
action of +/-1 requests a +/-0.5 rad target increment. These are target increments,
not guaranteed displacement per step or a physical velocity limit. Actual motion
depends on tracking dynamics, joint/actuator limits and contact.

Use the action range deliberately instead of globally restricting all motion
to a small fraction of it. For example, a clip of 0.12 limits each requested
position increment to 6 mm even when the arm is moving freely. Increasing the
clip cannot resolve a collision or a joint-limit stall.

- Separate free-space travel, final approach, contact manipulation and settling.
  Use faster commands with adequate clearance, then decelerate near the cube
  and during contact. Do not use maximum commands blindly while gripping.
- Terminate motion when measured position/orientation error and residual motion
  satisfy your tolerances; use a short confirmation window when needed. Avoid
  long fixed step loops that keep running after reaching the target.
- Monitor progress over a bounded window. If error stops decreasing, a gripper
  misses the cube, or a joint approaches its limit, stop and revise the pose,
  grasp or route instead of spending hundreds of steps on the same command.
- Bound retries and settling intervals. Verify contact and support before
  releasing the other hand. If the cube has dropped, use the rescue API.
- Drive turns with measured progress and alignment feedback. Excessively dense
  waypoint schedules and fixed waits can consume the budget without improving
  contact stability; do not accelerate beyond what reliable tracking permits.

Log steps spent on travel, settling, turning, handoffs and failed retries so you
can identify waste. Preserve grasp stability and minimize rotations; efficient
motion means fewer unnecessary steps, not just higher command amplitudes.
