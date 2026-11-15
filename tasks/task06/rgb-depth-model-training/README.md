# Task 06 / rgb-depth-model-training: learned pose estimation (20M-parameter limit)

Estimate the planar pose of an unlabeled T, C or F block from a fixed camera
and identify its shape; a learned model, trained from scratch in this container. See [instruction.md](instruction.md)
for the contract and the family [README](../README.md) for the scoring.

The agent container runs Python 3.12 with MuJoCo 3.5, numpy and scipy on
16 CPUs, 6144 MB RAM, 1 NVIDIA GPU, with a
7200-second session. The workspace holds clients for the
root-owned renderer and the design diagnostic; meshes, shape identity, exact
labels, seeds and grading code stay private. The deliverable is `model.pt` (TorchScript);
the gate is that the TorchScript module loads within the 20M-element limit.

## Running it

~~~bash
make task06-assets
harbor run -p tasks/task06/rgb-depth-model-training -a oracle
rlebench run task06/rgb-depth-model-training -a <agent> -m <model>  # adds the agent's install hosts
~~~
