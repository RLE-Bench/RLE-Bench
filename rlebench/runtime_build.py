"""Stage selected Harbor cells and their shared images without importing a simulator."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RUNTIME = ("__init__.py", "client.py", "protocol.py", "capabilities.py")
TABLETOP = (("01-tower-max-height", "TowerMaxHeight"), ("02-cantilever-overhang", "CantileverOverhang"),
            ("03-balance-coins", "BalanceCoins"), ("04-rubik-cube", "RubikCube"),
            ("05-hidden-center-of-mass", "HiddenCOM"))


def load(family):
    path = ROOT / "tasks" / family / "harness/config.py"
    spec = importlib.util.spec_from_file_location("build_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source, target):
    shutil.copytree(source, target, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def stage(family, level="L1", variant=""):
    here = ROOT / "tasks" / family
    context = here / "images" / level if family == "task01" else here / (f"image-{variant}" if variant else "image")
    if context.exists():
        shutil.rmtree(context)
    public, private = context / "payload_agent", context / "payload_private"
    for payload in (public, private):
        copy_file(ROOT / "rlebench/__init__.py", payload / "rlebench/__init__.py")
    for name in PUBLIC_RUNTIME:
        copy_file(ROOT / "rlebench/runtime" / name, public / "rlebench/runtime" / name)
    copy_tree(ROOT / "rlebench/runtime", private / "rlebench/runtime")
    for name in ("__init__.py", "media.py"):
        copy_file(ROOT / "rlebench/core" / name, private / "rlebench/core" / name)
    copy_tree(here / "harness", private / "harness")
    for name in ("__init__.py", "client.py"):
        copy_file(here / "harness" / name, public / "harness" / name)
    if family == "task01" and level in ("L2", "L3"):
        copy_tree(here / "harness/skills", public / "harness/skills")
        for name in ("obs.py", "perception_protocol.py"):
            copy_file(here / "harness" / name, public / "harness" / name)
    if family == "task03":
        copy_tree(here / "tabletop", private / "harness/tabletop")
        copy_file(here / "harness/obs.py", public / "harness/obs.py")
        if variant == "pocket":
            copy_file(here / "_template/cube_recovery.py", public / "cube_recovery.py")
            copy_file(here / "_template/pocket_manual.md", private / "HARNESS_MANUAL.md")
    for name in ("rlebench_ro_assets.py", "verify_assets.py"):
        copy_file(ROOT / "sim/robocasa/base" / name, private / name)
    template = here / "_template/image"
    for file in template.iterdir():
        copy_file(file, context / file.name)
    copy_file(ROOT / "sim/perception/requirements.lock", context / "requirements.lock")
    # Model layers are shared, but only task01 L2/L3 use them.
    pins = dict(line.split("=", 1) for line in (ROOT / "sim/robocasa/pins.env").read_text().splitlines() if line and not line.startswith("#"))
    text = (context / "Dockerfile").read_text().replace("@@LEVEL@@", level).replace("@@FAMILY@@", family)
    text = text.replace("@@SIM_IMAGE@@", pins["ROBOCASA_SIM_IMAGE"])
    text = text.replace("@@MODELS@@", "models-on" if family == "task01" and level != "L1" else "models-off")
    (context / "Dockerfile").write_text(text)
    return context


def image_tag(family, level="L1", variant=""):
    suffix = "-"+level.lower() if family == "task01" else "-"+variant if variant else ""
    return f"rlebench-{family}{suffix}-agent:dev"


def steps_for(family, count):
    if family == "task01":
        return [("develop", 32400)]
    if family == "task02":
        return [("develop", 28800)] + [(f"eval_{i+1:02d}", 3600) for i in range(count)]
    return []


def task_toml(family, slug, task, level, variant, count):
    steps = steps_for(family, count)
    head = 'schema_version = "1.4"\nartifacts = ["/workspace"]\n'
    if steps:
        head += 'multi_step_reward_strategy = "final"\n'
    head += f'''\n[task]
name = "rlebench/{family}-{level}-{slug}"
version = "1.1.0"
description = "Metered robot control through a private simulator service."
[metadata]
category = "robotics-control"
[agent]
user = "agent"
timeout_sec = {3600 if variant == "hidden-com" else 32400}.0
network_mode = "allowlist"
allowed_hosts = []
[environment]
docker_image = "{image_tag(family, level, variant)}"
cpus = 8
memory_mb = 24576
gpus = 1
network_mode = "public"
[environment.env]
RLEBENCH_TASK = "{task}"
RLEBENCH_LEVEL = "{level}"
RLEBENCH_TIMEOUT_MULT = "${{RLEBENCH_TIMEOUT_MULT:-1.0}}"
'''
    if family == "task02":
        head += f'RLEBENCH_GROUP = "{slug}"\n'
    if family == "task01":
        head += 'RLEBENCH_SESSION_SECONDS = "${RLEBENCH_SESSION_SECONDS:-32400}"\n'
    if family != "task03":
        head += 'RLEBENCH_MEDIA = "${RLEBENCH_MEDIA:-false}"\n'
        budget = load("task01").INTERACTION_STEPS if family == "task01" else 75000
        horizon = 1000 if family == "task01" else 5000
        head += f'RLEBENCH_INTERACTION_STEPS = "${{RLEBENCH_INTERACTION_STEPS:-{budget}}}"\n'
        head += f'RLEBENCH_MAX_STEPS_PER_TRIAL = "${{RLEBENCH_MAX_STEPS_PER_TRIAL:-{horizon}}}"\n'
    head += '''[environment.healthcheck]
command = "/opt/healthcheck.sh"
interval_sec = 2.0
timeout_sec = 5.0
start_period_sec = 300.0
retries = 10
[verifier]
user = "root"
timeout_sec = 300.0
network_mode = "no-network"
'''
    if not steps:
        head += '''[[verifier.collect]]
command = "/opt/control.sh attempt"
user = "root"
timeout_sec = 180.0
'''
    for name, seconds in steps:
        head += f'''\n[[steps]]
name = "{name}"
[steps.agent]
user = "agent"
timeout_sec = {seconds}.0
[steps.verifier]
user = "root"
timeout_sec = 300.0
env = {{ RLEBENCH_STEP = "{name}" }}
[[steps.verifier.collect]]
command = "/opt/control.sh {name}"
user = "root"
timeout_sec = 180.0
'''
    return head


def instruction(family, phase, task, level):
    template = ROOT / "tasks" / family / "_template/steps"
    if family == "task02":
        return (template / ("develop" if phase == "develop" else "eval_01") / "instruction.md").read_text()
    text = (template / phase / "instruction.md.in").read_text()
    fragment = (template / "develop" / f"harness_{level}.md").read_text().rstrip("\n")
    return (text.replace("@@ROBOCASA_TASK@@", task)
            .replace("@@GOAL@@", dict(load(family).SUBTASKS)[task])
            .replace("@@HARNESS_SECTION@@", fragment))


def tabletop_instruction(task):
    template = ROOT / "tasks/task03/_template"
    if task in ("RubikCube", "HiddenCOM"):
        name = "pocket.md" if task == "RubikCube" else "hidden_com.md"
        budget = "50,000" if task == "RubikCube" else "12,000"
        return (template / name).read_text().replace("@@INTERACTION_STEPS@@", budget)
    details = json.loads((template / "tasks.json").read_text())[task]
    return ((template / "instruction.md.in").read_text()
            .replace("@@INTERACTION_STEPS@@", "50,000")
            .replace("@@GOAL@@", details["goal"])
            .replace("@@TASK_NOTES@@", details["notes"]))


def emit_cell(family, slug, task, level, variant, count):
    here = ROOT / "tasks" / family
    destination = here / level / slug if family == "task01" else here / slug
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    (destination / "task.toml").write_text(task_toml(family, slug, task, level, variant, count))
    env = destination / "environment"
    env.mkdir()
    volumes = '' if family == "task03" else '''    volumes:
      - ${ROBOCASA_ASSET_DIR:?set ROBOCASA_ASSET_DIR}:/opt/src/robocasa/robocasa/models/assets:ro
'''
    (env / "docker-compose.yaml").write_text('''services:
  main:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['${RLEBENCH_GPU:-0}']
              capabilities: [gpu]
'''+volumes)
    testdir = destination / "tests"
    testdir.mkdir()
    (testdir / "test.sh").write_text('''#!/bin/sh
set -u
exec /opt/verify.sh "${RLEBENCH_STEP:-attempt}"
''')
    (testdir / "test.sh").chmod(0o755)
    steps = steps_for(family, count)
    for name, _ in steps:
        step = destination / "steps" / name
        (step / "workdir").mkdir(parents=True)
        (step / "instruction.md").write_text(instruction(family, name, task, level))
        (step / "workdir/setup.sh").write_text('''#!/bin/sh
rm -f /logs/verifier/reward.json /logs/verifier/diagnosis.json 2>/dev/null || true
rm -f /workspace/setup.sh
exit 0
''')
        solution = step / "solution"
        solution.mkdir()
        code = ('sim.reset(); sim.step([0.]*12, ObsSpec(cameras=())); sim.end_development()'
                if name == "develop" else
                'sim.finish_trial()' if family == "task02" else
                'sim.task_info()\n    while sim.status()["phase"] != "finished":\n        sim.finish_trial()\n        sim.next_trial()')
        if family == "task01":
            code += '\n    while sim.status()["phase"] != "finished":\n        sim.finish_trial()\n        sim.next_trial()'
        (solution / "solve.sh").write_text('#!/bin/sh\ncd /workspace\npython - <<\'PY\'\nfrom harness.client import SimClient, ObsSpec\nwith SimClient() as sim:\n    '+code+'\nPY\n')
        (solution / "solve.sh").chmod(0o755)
    if not steps:
        (destination / "instruction.md").write_text(tabletop_instruction(task))
        solution_source = here / "_template" / ("hidden_com_solution" if variant == "hidden-com" else "solution")
        if variant != "pocket" and solution_source.exists():
            copy_tree(solution_source, destination / "solution")
            if not variant:
                actions = here / "_template/solution/actions" / (task+".npz")
                copy_file(actions, destination / "solution/actions.npz")
                shutil.rmtree(destination / "solution/actions", ignore_errors=True)
    else:
        (destination / "instruction.md").write_text(instruction(family, "develop", task, level))
    return destination


def check(context):
    public = context / "payload_agent"
    allowed = {"__init__.py", *(f"runtime/{name}" for name in PUBLIC_RUNTIME)}
    actual = {p.relative_to(public / "rlebench").as_posix()
              for p in (public / "rlebench").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    if actual != allowed:
        raise ValueError("unexpected shared code in the public payload")
    if (public / "harness/runtime").exists():
        raise ValueError("obsolete family runtime in the public payload")
    forbidden = {"task.py", "adapter.py", "backend.py", "compat.py", "stages.py", "config.py", "metrics.py"}
    if any(p.name in forbidden for p in public.rglob("*.py")):
        raise ValueError("private family code entered public payload")


def main(family=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("task01", "task02", "task03"), default=family)
    parser.add_argument("--emit", default=None)
    parser.add_argument("--emit-all", action="store_true")
    parser.add_argument("--cell")
    parser.add_argument("--level", default="L1")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    family = args.family
    if not family:
        parser.error("--family is required")
    if args.check:
        contexts = list((ROOT / "tasks" / family).glob("**/payload_agent"))
        if not contexts:
            raise SystemExit("no generated payloads")
        for context in contexts:
            check(context.parent)
        print(f"{family}: {len(contexts)} public/private boundaries checked")
        return
    C = load(family) if family != "task03" else None
    cells = ([(C.subtask_slug(i+1, name), name) for i, (name, _) in enumerate(C.SUBTASKS)]
             if family == "task01" else [(slug, "") for slug in C.SPLITS] if family == "task02" else list(TABLETOP))
    if args.list:
        print('\n'.join(slug for slug, _ in cells))
        return
    selected = args.cell or (args.emit if family != "task01" else None)
    if selected:
        cells = [(s, t) for s, t in cells if s == selected]
        if not cells:
            parser.error("unknown cell")
    elif not args.emit_all and args.emit is None:
        cells = cells[:1]
    levels = (list(C.LEVELS) if args.emit_all else [args.emit or args.level]) if family == "task01" else ["L1"]
    contexts = {}
    for level in levels:
        for slug, task in cells:
            variant = "pocket" if task == "RubikCube" else "hidden-com" if task == "HiddenCOM" else ""
            key = (level, variant)
            if key not in contexts:
                contexts[key] = stage(family, level, variant)
                check(contexts[key])
            destination = emit_cell(family, slug, task, level, variant, C.TRIALS_PER_TASK if family == "task02" else 5)
            print(destination.relative_to(ROOT))
    if args.build:
        for (level, variant), context in contexts.items():
            command = ["docker", "build", "-t", image_tag(family, level, variant),
                       "--build-arg", "AGENT_UID="+os.environ.get("AGENT_UID", "1000")]
            if family == "task01" and level != "L1":
                source = Path(os.environ.get("RLEBENCH_PERCEPTION_SOURCE") or
                              os.environ.get("RLEBENCH_PERCEPTION_VENDOR") or
                              ROOT / "third_party/perception").resolve()
                result = subprocess.run(
                    [str(ROOT / "sim/perception/perception.sh"), "setup"],
                    env={**os.environ, "RLEBENCH_PERCEPTION_VENDOR": str(source)},
                )
                if result.returncode:
                    raise SystemExit(result.returncode)
                command += ["--build-context", "perception="+str(source)]
                model_pins = dict(line.split("=", 1) for line in (ROOT / "sim/perception/pins.env").read_text().splitlines() if line and not line.startswith("#"))
                for name in ("TORCH_VERSION", "SAM3_SHA", "CGN_SHA"):
                    command += ["--build-arg", name+"="+model_pins[name]]
            subprocess.run([*command, str(context)], check=True)


if __name__ == "__main__":
    main()
