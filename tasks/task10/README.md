# Task 10: Fault-Adaptive Arm Control

This hard CPU-only Harbor task evaluates safe system identification and adaptive Cartesian/contact control under hidden coupled actuator faults. Run `python tasks/task10/build_assets.py` before building the two separate images.

The agent receives public specifications, a nominal Panda model and explicit practice profiles. Hidden seeds, fault sampling, scoring thresholds and the observation-only golden controller are staged only into the verifier image. The controller itself is re-imported under uid 65534 in a submission-only subprocess.

Host validation lives in `tests/test_task10.py`; development calibration and difficulty evidence live under `dev/` and never ship into either image.


## Calibrated references

The final separate-verifier Oracle scores 0.757557 across eight seeds and three bands with no safety violations. The nominal controller is safety-gated to 0.100 and the zero controller scores 0.000. Full deterministic reports are retained under dev and regenerated through the host test and container Oracle paths.
