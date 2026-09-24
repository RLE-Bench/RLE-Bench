import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os
sys.path.insert(0, str(ROOT / "tasks" / os.environ.get("RLEBENCH_TEST_TASK", "task01")))

if os.environ.get("RLEBENCH_TEST_TASK") == "task03":
    import harness
    harness.__path__.append(str(ROOT / "tasks/task03"))
