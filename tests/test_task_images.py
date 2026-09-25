"""Dependency pins agree across the host, task images and generators."""
from __future__ import annotations

import glob
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


_PIN_RE = re.compile(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9+.]+)")

_PIN_DOCKERFILES = sorted(
    os.path.relpath(p, REPO)
    for pattern in (
        "tasks/task08/*/Dockerfile",
        "tasks/task09/*/Dockerfile",
        "tasks/task07/*/Dockerfile",
        "tasks/task06/*/*/Dockerfile",
    )
    for p in glob.glob(os.path.join(REPO, pattern))
)


def _requirements_pins() -> dict[str, str]:
    pins = {}
    for line in open(os.path.join(REPO, "requirements.txt")):
        m = _PIN_RE.match(line.strip())
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def _file_pins(relpath: str) -> list[tuple[str, str]]:
    return [(n.lower(), v) for n, v in _PIN_RE.findall(open(os.path.join(REPO, relpath)).read())]


def _generator_pins() -> list[tuple[str, str]]:
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("task06_build_assets", os.path.join(REPO, "tasks", "task06", "build_assets.py"))
    g = _ilu.module_from_spec(spec); spec.loader.exec_module(g)

    tokens = " ".join(g.PINS_COMMON) + f" {g.PIN_OPENCV} {g.PIN_SKLEARN} {g.PIN_TORCH_CPU} {g.PIN_TORCH_CUDA}"
    return [(n.lower(), v) for n, v in _PIN_RE.findall(tokens)]


def test_core_pins_match_requirements():
    """Every pip pin of a requirements.txt package, in every image of the
    family and in the task06 generator, matches it -- torch by base version,
    with only the +cpu / +cu128 locals in use today."""
    canon = _requirements_pins()
    assert canon, "requirements.txt has no pins?"
    torch_base = canon["torch"].split("+")[0]
    problems = []
    sources = [(p, _file_pins(p)) for p in _PIN_DOCKERFILES]
    sources.append(("tasks/task06/build_assets.py", _generator_pins()))
    for src, pins in sources:
        assert pins, f"{src}: no pins found (pattern drift?)"
        for name, version in pins:
            if name == "torch":
                base, _, local = version.partition("+")
                if base != torch_base or local not in ("cpu", "cu128"):
                    problems.append(f"{src}: torch=={version} != {torch_base}+(cpu|cu128)")
            elif name in canon and version != canon[name]:
                problems.append(f"{src}: {name}=={version} != requirements {canon[name]}")
    assert not problems, "\n".join(problems)


def test_orphan_pins_agree_everywhere():
    """Packages the images pin but requirements.txt does not (today:
    opencv-python-headless, scikit-learn) must carry one version everywhere,
    including the generator constants -- mutual consistency, no second table."""
    canon = _requirements_pins()
    seen: dict[str, dict[str, list[str]]] = {}
    sources = [(p, _file_pins(p)) for p in _PIN_DOCKERFILES]
    sources.append(("tasks/task06/build_assets.py", _generator_pins()))
    for src, pins in sources:
        for name, version in pins:
            if name in canon or name == "torch":
                continue
            seen.setdefault(name, {}).setdefault(version, []).append(src)
    conflicts = {n: v for n, v in seen.items() if len(v) > 1}
    assert not conflicts, f"orphan pins disagree across images: {conflicts}"


def test_task01_single_session_configuration():
    import tomllib
    from rlebench.runtime_build import task_toml, instruction
    config = tomllib.loads(task_toml('task01', '01-open-fridge', 'OpenFridge', 'L1', '', 5))
    assert config['task']['version'] == '1.2.0'
    assert config['agent']['timeout_sec'] == 32400
    assert [s['name'] for s in config['steps']] == ['develop']
    assert config['steps'][0]['agent']['timeout_sec'] == 32400
    assert config['steps'][0]['verifier']['collect'][0]['command'] == '/opt/control.sh develop'
    for level in ('L1', 'L2', 'L3'):
        prompt = instruction('task01', 'develop', 'OpenFridge', level)
        assert 'sim.end_development()' in prompt and 'sim.next_trial()' in prompt
        assert '@@' not in prompt
    transfer = tomllib.loads(task_toml('task02', '01-washing-dishes', '', 'L1', '', 5))
    assert len(transfer['steps']) == 6


def test_shared_runtime_payload_isolation(tmp_path, monkeypatch):
    import subprocess
    import sys
    import pytest
    from rlebench import runtime_build as build

    source = build.ROOT
    for relative in ('rlebench/__init__.py', 'sim/robocasa/pins.env',
                     'sim/robocasa/base/rlebench_ro_assets.py', 'sim/robocasa/base/verify_assets.py',
                     'sim/perception/requirements.lock'):
        build.copy_file(source / relative, tmp_path / relative)
    build.copy_tree(source / 'rlebench/runtime', tmp_path / 'rlebench/runtime')
    for family in ('task01', 'task02', 'task03'):
        for part in ('harness', '_template'):
            build.copy_tree(source / 'tasks' / family / part, tmp_path / 'tasks' / family / part)
        assert not (source / 'tasks' / family / 'harness/runtime').exists()
    build.copy_tree(source / 'tasks/task03/tabletop', tmp_path / 'tasks/task03/tabletop')
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    cases = [('task01', level, '') for level in ('L1', 'L2', 'L3')]
    cases += [('task02', 'L1', ''), *[('task03', 'L1', variant) for variant in ('', 'pocket', 'hidden-com')]]
    for family, level, variant in cases:
        context = build.stage(family, level, variant)
        build.check(context)
        public, private = context / 'payload_agent', context / 'payload_private'
        for payload in (public, private):
            assert not (payload / 'harness/runtime').exists()
        code = '''import importlib, importlib.util, sys
sys.path.insert(0, sys.argv[1])
from harness.client import SimClient
from rlebench.runtime.client import SimClient as SharedClient
assert SimClient is SharedClient
for name in ('engine', 'server', 'worker', 'store', 'scoring', 'verify', 'control', 'handoff', 'process', 'launcher', 'media'):
    try: importlib.import_module('rlebench.runtime.' + name)
    except ModuleNotFoundError: pass
    else: raise AssertionError(name + ' exposed')
for name in ('rlebench.cli', 'harness.task', 'harness.config', 'harness.adapter'):
    assert importlib.util.find_spec(name) is None, name
'''
        subprocess.run([sys.executable, '-I', '-S', '-c', code, str(public)], check=True)
        code = '''import sys
sys.path.insert(0, sys.argv[1])
from rlebench.runtime.engine import Engine
from rlebench.runtime.server import Server
from harness.client import SimClient
'''
        subprocess.run([sys.executable, '-I', '-S', '-c', code, str(private)], check=True)
        for unexpected in ('runtime/engine.py', 'cli.py', 'runtime/hidden.json'):
            path = public / 'rlebench' / unexpected
            path.write_text('private')
            with pytest.raises(ValueError, match='unexpected shared code'):
                build.check(context)
            path.unlink()
