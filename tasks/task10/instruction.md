# Fault-Adaptive Arm Control

Ship one controller for a torque-controlled seven-joint Franka Panda whose sensors and actuators differ from the supplied nominal MuJoCo model. A bounded combination of encoder zero offsets, actuator gain error, viscous friction, a weakened joint, adjacent-joint actuator cross-talk, zero-to-three control ticks of command delay and an unknown payload is fixed for each episode. In the recovery band, one actuator weakens again during execution and the arm receives a bounded impulse.

The verifier first provides 160 calibration ticks. It then resets the physical arm to the published home pose while preserving the controller object and its estimates. Evaluation starts with an announced 80-tick stabilization segment, followed by one of three bands: free-space Cartesian tracking, tracking into and along a compliant surface at a requested normal force, or recovery after an impulse and mid-run actuator derating.

Create `/logs/artifacts/controller.py`. The seed passed to reset is the fixed public value 0 and is independent of hidden scenario seeds. It must export `make_controller()`, returning an object with `reset(context: dict, seed: int)` and `act(observation: dict)`. Return exactly seven finite normalized torque commands in `[-1, 1]`; the verifier performs the final clipping and torque scaling.

`context` contains the path to a read-only nominal MJB, joint and actuator names, the end-effector site name, home angles, torque limits, timestep, calibration and evaluation lengths, all public fault bounds and the scenario name. The MJB is a private copy: changing it cannot change verifier physics.

Each observation contains `step`, offset joint positions `qpos`, joint velocities `qvel`, the last commanded torque, quantized noisy motor current, independently measured end-effector position, filtered normal contact force and `target`. The target contains Cartesian position, desired normal force and a phase equal to `calibration` or `evaluation`. Motor current exposes actuator response but not friction torque, payload or sampled parameters. End-effector position is sensor output, not direct access to MuJoCo state.

The submission runs in a fresh restricted subprocess as uid 65534. It cannot read verifier files, inherit `PYTHONPATH`, fork child processes or mutate parent simulation state. Only NumPy, SciPy and MuJoCo are available; network access is disabled. Startup/reset has a 30-second limit, each action has a 2-second limit, and `controller.py` must be a regular file no larger than 1 MiB.

Scoring uses only verifier instrumentation. Free-space score is based on Cartesian RMSE. Contact score combines Cartesian RMSE, filtered normal-force RMSE and bounded peak force, and requires at least 30 scored ticks above 2 N; never touching earns zero contact score. Recovery combines post-disturbance RMSE with time to regain a 45 mm error tube. Repeated invalid actions, hard joint-limit violations or five consecutive over-force ticks cap total reward at 0.10.

Use the public runner on three explicit practice profiles:

```bash
python3 /workspace/dev_runner.py my_controller.py --profile mild --scenario free_space
python3 /workspace/dev_runner.py my_controller.py --profile coupled --scenario contact
python3 /workspace/dev_runner.py my_controller.py --profile stress --scenario recovery
```

The practice profiles are deliberately explicit and are not generated from hidden seeds. Evaluation seeds, sampled faults, thresholds, scorer and reference controller are absent from the agent image.
