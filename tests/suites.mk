# Host-side pytest modules owned by each Harbor task family.
# Cross-cutting tests run once through `make test-host`.

TASK_IDS := task01 task02 task03 task04 task05 task06 rgb-only rgb-depth rgb-depth-model-training method-agnostic task07 task08 task09 task10

HOST_TESTS := tests/test_task_images.py tests/test_rlebench_argv.py tests/test_rlebench_lint.py \
	tests/test_rlebench_summarize.py tests/test_antigravity_agents.py \
	tests/test_camera_intrinsics.py tests/test_core_media.py tests/test_rlebench_view.py

TASK_TESTS_task01 := $(wildcard tests/test_speedrun_*.py)
TASK_TESTS_task02 := $(wildcard tests/test_toolsmith_*.py)

# task03 owns the tabletop prefix, which is also why the harness-composition test
# is test_tabletop_harness.py and not test_speedrun_*: it belongs to this family,
# not to the shared harness suite task01 runs.
TASK_TESTS_task03 := $(wildcard tests/test_tabletop_*.py)

TASK_TESTS_task04 := $(wildcard tests/test_motiontrack_*.py)

TASK_TESTS_task05 := $(wildcard tests/test_nanovla_*.py) tests/test_rlebench_task05.py

TASK_TESTS_task06 := $(wildcard tests/test_percept_*.py)
TASK_TESTS_rgb-only = $(TASK_TESTS_task06)
TASK_TESTS_rgb-depth = $(TASK_TESTS_task06)
TASK_TESTS_rgb-depth-model-training = $(TASK_TESTS_task06)
TASK_TESTS_method-agnostic = $(TASK_TESTS_task06)

TASK_TESTS_task07 := $(wildcard tests/test_binclear_*.py)

TASK_TESTS_task08 := $(wildcard tests/test_base_design_*.py)

TASK_TESTS_task09 := $(wildcard tests/test_gello_*.py)
TASK_TESTS_task10 := tests/test_task10.py

# The families whose mujoco pin needs its own venv; the rest run on PY.
PY_task04 := .venv-motiontrack/bin/python

# `make test-<family>-metrics` (fast pure-math checks) and `-golden` (the reference
# scores at its calibrated bar, degenerate submissions do not, isolation holds).
TASK_GOLDEN_task04  := tests/test_motiontrack_golden.py tests/test_motiontrack_antigaming.py tests/test_motiontrack_task.py
TASK_METRICS_task06 := tests/test_percept_scene.py tests/test_percept_oracle.py
TASK_GOLDEN_task06  := tests/test_percept_scoring.py tests/test_percept_antigaming.py
TASK_METRICS_task07 := tests/test_binclear_metrics.py tests/test_binclear_scene.py
TASK_GOLDEN_task07  := tests/test_binclear_golden.py tests/test_binclear_antigaming.py
TASK_METRICS_task08 := tests/test_base_design_metrics.py
TASK_GOLDEN_task08  := tests/test_base_design_golden.py
TASK_METRICS_task09 := tests/test_gello_metrics.py tests/test_gello_codesign_metrics.py tests/test_gello_scoring.py
TASK_GOLDEN_task09  := tests/test_gello_codesign_golden.py tests/test_gello_packaging.py tests/test_gello_codesign_tiers.py

TASK_METRICS_task10 := tests/test_task10.py
TASK_GOLDEN_task10 := tests/test_task10.py
