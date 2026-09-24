# Hidden center-of-mass development tools

The runnable single-stage task is emitted by `make task03-assets`.
See [the task README](../../README.md) for its interface,
private simulator boundary, build commands and Oracle run.

Render the shared scene using the pinned simulator environment:

```bash
.venv-robocasa/bin/python tasks/task03/dev/hidden_com/preview.py
```

`--quadrant A|B|C|D` selects the preview fixture; `--output DIR` overrides
`results/task03/hidden_com`. Outputs include three views, a comparison
sheet, portable XML/assets, and mass/COM and repeatability checks. Load the
`preview` keyframe to inspect the initial pose. XML contains private inertia
and is only for maintainers.

`tests/runtime/container_check.py --family task03 --variant hidden-com
--task HiddenCOM --report /tmp/hidden-com-check.json` checks the running image,
including private-file isolation as the agent user. Host-side physics checks
live in `tests/runtime/test_hidden_scene.py`.
