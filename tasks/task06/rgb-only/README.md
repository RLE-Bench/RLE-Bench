# Task 06 / rgb-only: RGB-only pose estimation

Estimate the planar pose of an unlabeled T, C or F block from a fixed camera
and identify its shape; any CPU method using RGB only. See [instruction.md](instruction.md)
for the contract and the family [README](../README.md) for the scoring.

The agent container runs Python 3.12 with MuJoCo 3.5, numpy and scipy on
4 CPUs, 6144 MB RAM, CPU only, with a
7200-second session. The workspace holds clients for the
root-owned renderer and the design diagnostic; meshes, shape identity, exact
labels, seeds and grading code stay private. The deliverable is `estimator.py`;
the gate is that the estimator loads and returns a pose.

## Running it

~~~bash
make task06-assets
harbor run -p tasks/task06/rgb-only -a oracle
rlebench run task06/rgb-only -a <agent> -m <model>  # adds the agent's install hosts
~~~
