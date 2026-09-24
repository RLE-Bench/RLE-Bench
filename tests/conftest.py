import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Family suites import their task's `harness` package. The name is shared
# across families on purpose (images import it uniformly), so exactly one
# family dir may sit on sys.path per process -- selected by RLEBENCH_TEST_TASK.
_FAMILY_DIRS = {
    "task01": "task01", "task02": "task02", "task03": "task03",
    "task04": "task04", "task05": "task05", "task06": "task06",
    "task07": "task07", "task08": "task08", "task09": "task09",
    "rgb-only": "task06", "rgb-depth": "task06",
    "rgb-depth-model-training": "task06", "method-agnostic": "task06",
}
_selected = os.environ.get("RLEBENCH_TEST_TASK")
if _selected in _FAMILY_DIRS:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "tasks", _FAMILY_DIRS[_selected]))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build missing family payloads before collection on a fresh checkout.
GENERATED_TASK_DIRS = {
    "rgb-only": ("environment/agent", "environment/private",
                "tests/harness/assets", "solution/payload"),
    "rgb-depth": ("environment/agent", "environment/private",
                "tests/harness/assets", "solution/payload"),
    "rgb-depth-model-training": ("environment/agent", "environment/private",
                "tests/harness/assets", "solution/payload"),
    "method-agnostic": ("environment/agent", "environment/private",
                "tests/harness/assets", "solution/payload"),
    "task07": ("environment/assets/assets/franka_emika_panda",
                "tests/assets/franka_emika_panda", "tests/harness"),
    "task08": ("environment/assets", "tests/harness", "tests/models",
                "solution/payload"),
    "task09": ("environment/assets", "tests/harness", "tests/models",
                "solution/payload"),
}

TASK06_SUBTASKS = frozenset((
    "rgb-only",
    "rgb-depth",
    "rgb-depth-model-training",
    "method-agnostic",
))


def pytest_sessionstart(session):
    """Build generated trees before collection imports their packages."""
    selected_task = os.environ.get("RLEBENCH_TEST_TASK")
    if not selected_task:
        return
    # task05 emits its matrix from _template/ + subtasks.toml rather than staging
    # payload trees, so it has no build_assets.py for the loop below to find.
    if selected_task == "task05" and not os.path.isfile(
            os.path.join(REPO, "tasks/task05/01-libero-open-design/task.toml")):
        subprocess.run([sys.executable, "tasks/task05/build_subtasks.py", "--emit-all"],
                       cwd=REPO, check=True)
    task_dirs = GENERATED_TASK_DIRS
    if selected_task == "task06":
        task_dirs = {
            task: dirs for task, dirs in GENERATED_TASK_DIRS.items()
            if task in TASK06_SUBTASKS
        }
    elif selected_task:
        task_dirs = ({selected_task: GENERATED_TASK_DIRS[selected_task]}
                     if selected_task in GENERATED_TASK_DIRS else {})

    built = set()
    for task, dirs in task_dirs.items():
        is_task06 = task in TASK06_SUBTASKS
        root = os.path.join(
            REPO, "tasks", "task06", task
        ) if is_task06 else os.path.join(REPO, "tasks", task)
        script = os.path.join(root, "build_assets.py")
        if not os.path.isfile(script) and not is_task06:
            continue
        if all(os.path.exists(os.path.join(root, d)) for d in dirs):
            continue
        missing = [d for d in dirs
                   if not os.path.exists(os.path.join(root, d))]
        if is_task06:
            script = os.path.join(REPO, "tasks", "task06", "build_assets.py")
        if script in built:
            continue
        source = os.path.relpath(script, REPO)
        print(f"\n[conftest] {task}: regenerating {missing} via {source}",
              flush=True)
        command = [sys.executable, script]
        subprocess.run(command, cwd=REPO, check=True)
        built.add(script)
