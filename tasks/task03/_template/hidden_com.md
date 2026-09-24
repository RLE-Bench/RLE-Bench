# Hidden center of mass

Use the Panda robot to identify the hidden ballast quadrant in three sealed boxes.
The boxes look identical but have different masses, ballast placements, and inertias.
Determine each answer correctly using as few control steps as possible. You cannot
open the boxes or reset an experiment.

Identify quadrants by the A/B/C/D markings on the box.

Use `harness.client.HiddenCOMClient` from Python in `/workspace`:

```python
from harness.client import HiddenCOMClient
from PIL import Image

sim = HiddenCOMClient()
obs = sim.observe()
Image.fromarray(obs["images"]["top"]).save("/workspace/top.png")
# Interact until ready, then submit your chosen quadrant:
# result = sim.submit("A")
```

Submit once per box with `sim.submit("A")` (or B, C, D). Answers are final;
submissions acknowledge receipt without correctness feedback. The first two
submissions load the next box: call `sim.observe()` before acting again. After
the third submission returns `done=True`, end your turn. Answer files and chat
text do not submit answers.

Each box permits **@@INTERACTION_STEPS@@ control steps at 20 Hz**, with 1–200 steps per movement
call. Batches exceeding the remaining budget are rejected without movement.
At the cap you can still observe and submit. Reconnecting preserves progress.

If `ended` is `sim_error`, that box is closed without answer credit. Submit any
valid quadrant to advance to the next box, then observe before acting.

API documentation and scene dimensions are in the agent-visible
`/opt/rlebench/harness/client.py` (also available through Python `help()`).
