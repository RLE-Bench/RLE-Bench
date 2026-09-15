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
        "tasks/task10/*/Dockerfile",
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
