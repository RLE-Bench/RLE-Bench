"""Run the six-step Harbor protocol while killing a worker and then its broker."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--gpu", default="4")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    name = "runtime-fault-"+uuid.uuid4().hex[:8]
    cell = root / "reports" / name
    shutil.copytree(root / "tasks/task02/06-setting-the-table", cell)
    develop = cell / "steps/develop/solution/solve.sh"
    develop.write_text(develop.read_text().replace("cd /workspace", "cd /workspace\nprintf frozen > agent_harness/MANUAL.md"))
    for step in cell.glob("steps/eval_*/solution/solve.sh"):
        step.write_text(step.read_text().replace("cd /workspace", "cd /workspace\ntest \"$(cat agent_harness/MANUAL.md)\" = frozen || exit 1"))
    for index in (1, 2):
        code = '''#!/bin/sh
cd /workspace
python - <<'PY'
from pathlib import Path
import time
from harness.client import SimClient, RemoteError
assert Path('/workspace/agent_harness/MANUAL.md').read_text()=='frozen'
Path('/workspace/agent_harness/MANUAL.md').write_text('evaluation edit')
with SimClient() as sim:
    sim.task_info()
    sim.step([0.]*12)
Path('/workspace/fault_ready').write_text('INDEX')
for _ in range(120):
    try:
        with SimClient() as sim:
            if sim.status()['ended']=='sim_error': break
    except (OSError, RemoteError): pass
    time.sleep(.5)
else: raise AssertionError('fault not observed')
PY
'''.replace("INDEX", str(index))
        (cell / f"steps/eval_{index:02d}/solution/solve.sh").write_text(code)
    env = dict(os.environ, ROBOCASA_ASSET_DIR=str(args.assets.resolve()), RLEBENCH_GPU=args.gpu)
    log_path = root / "reports/harbor-faults.log"
    with log_path.open("w") as log:
        run = subprocess.Popen(["harbor", "run", "-p", str(cell), "-a", "oracle",
             "--override-gpus", "0", "--yes", "--jobs-dir", str(root/"jobs"),
             "--job-name", name], env=env, stdout=log, stderr=subprocess.STDOUT)
        injected = set()
        deadline = time.monotonic()+360
        try:
            while run.poll() is None and time.monotonic()<deadline:
                containers = subprocess.check_output(["docker","ps","--filter","name="+name,
                                      "--format","{{.Names}}"],text=True).splitlines()
                for container in containers:
                    ready = subprocess.run(["docker","exec",container,"cat","/workspace/fault_ready"],
                                           capture_output=True,text=True).stdout.strip()
                    if ready not in ("1","2") or ready in injected:
                        continue
                    module = "harness.runtime.worker" if ready=="1" else "harness.runtime.server"
                    kill = """from pathlib import Path
import os,signal
for path in Path('/proc').glob('[0-9]*/cmdline'):
    try: args=path.read_bytes().split(b'\\0')
    except (FileNotFoundError,ProcessLookupError): continue
    if MODULE in args: os.kill(int(path.parent.name),signal.SIGKILL)
""".replace("MODULE", repr(module.encode()))
                    subprocess.run(["docker","exec","-i","--user","root",container,"python","-"],
                                   input=kill,text=True,check=True)
                    injected.add(ready)
                time.sleep(.5)
            if run.poll() is None:
                raise TimeoutError("Harbor fault check exceeded six minutes")
            assert run.returncode==0 and injected=={"1","2"}, (injected, log_path)
            job = root / "jobs" / name
            rewards = list(job.glob("*/steps/eval_05/verifier/reward.json"))
            assert len(rewards)==1, list(job.rglob("reward.json"))
            reward = json.loads(rewards[0].read_text())
            assert reward["trials_recorded"]==5 and reward["infrastructure_failures"]==2, reward
            (root/"reports/harbor-faults.json").write_text(json.dumps(dict(
                ok=True, job=name, faults=["worker SIGKILL in eval_01", "broker SIGKILL in eval_02"],
                final_reward=reward),indent=2)+"\n")
        finally:
            if run.poll() is None:
                run.terminate()
                run.wait(timeout=30)


if __name__ == "__main__":
    main()
