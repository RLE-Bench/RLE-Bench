**You are given the object and fixture poses**, under `priv_` keys that arrive with every
observation:

| key | what |
|---|---|
| `priv_target` | which object this episode is about, and what it is resting on |
| `priv_objects` | **every** object in the scene by name, with world `pos` and `quat` |
| `priv_fixtures` | fixtures (drawer, sink, cabinet, stove) with pose **and extents** |
| `priv_grasp` | `grasping`, `grasped_objects`, and per-finger contact flags |
| `priv_object_state` | the simulator's own raw pose keys, verbatim |

```python
obs = sim.reset()["obs"]
target = obs["priv_target"]["name"]              # e.g. "obj"
where  = obs["priv_objects"][target]["pos"]      # world frame, metres
held   = obs["priv_grasp"]["grasping"]
```

Poses are in the **world** frame. Actions are in the **base** frame, and
`robot0_base_pos` / `robot0_base_quat` are in the observation — converting between them
is yours to do. Quaternions are **xyzw**, the same order as every `robot0_*_quat` key,
so one convention covers the whole observation.

This is localisation, never task understanding: it does not say which object matters
beyond `priv_target`, and it does not say whether you have succeeded — that is only ever
the environment's own predicate. Every step you take is still charged.

The cameras are still there. Use them where they are easier.

### You also have the harness library

**Its manual is at `/opt/HARNESS_MANUAL.md`.** The same library the L2 harness ships, so
you are not writing servo loops on top of the poses you were handed:

```python
from harness.skills import transforms
from harness.skills import reach, grasp, lift, settle, move_base

target_base = transforms.world_to_base(obs, obs["priv_objects"]["obj"]["pos"])
reach(sim, target_base, gripper=-1.0)
settle(sim, gripper=-1.0); grasp(sim); lift(sim, 0.15)
```

`transforms.world_to_base` is the one you will reach for first — the poses are world and
your actions are not. `camera`, `geometry` and `perception` (SAM3 segmentation,
Contact-GraspNet proposals, both free) are there too, for the parts of the scene the
privileged keys do not describe — how an object is shaped, where its handle is, what a
good grasp on it would be.

Skills are ordinary code in **your** process over the same metered socket: their steps
cost what your own would, and they see exactly what you see.
