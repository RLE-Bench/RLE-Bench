"""Regression tests for task09's agent/verifier boundary."""

import inspect
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from harness import (balance_codesign_public, codesign_oracle,
                             codesign_scorer, sandbox, scenarios,
                             scenarios_public)


PRIVATE_SCENARIO_NAMES = {
    "GelloEnvelope",
    "compose_instance",
    "fixed_hold_configs",
    "hold_configs",
    "sample_workspace",
    "teleop_path",
    "validate_model_xml",
}


def test_public_scenarios_are_nominal_only():
    assert list(inspect.signature(scenarios_public.compose_lead).parameters) == [
        "lead_xml_path"
    ]
    for name in PRIVATE_SCENARIO_NAMES:
        assert not hasattr(scenarios_public, name)


def test_public_workspace_grid_matches_verifier():
    np.testing.assert_allclose(
        scenarios_public.workspace_grid(), scenarios.workspace_grid())


def test_public_balance_module_omits_simulation_scenarios():
    assert hasattr(balance_codesign_public, "balance_quality")
    assert hasattr(balance_codesign_public, "operator_effort")
    assert not hasattr(balance_codesign_public, "hold_sim")
    assert not hasattr(balance_codesign_public, "sag")


def test_model_xml_rejects_include_and_external_paths(tmp_path):
    sref = Path(codesign_oracle.CODESIGN_MODELS) / "franka" / "lead_sref.xml"
    shutil.copytree(sref.parent / "meshes", tmp_path / "meshes")
    for xml in (
        f'<mujoco><include file="{sref}"/></mujoco>',
        f'<mujoco><compiler meshdir="{sref.parent / "meshes"}"/></mujoco>',
        '<mujoco><compiler meshdir="../outside"/></mujoco>',
        '<mujoco><asset><mesh name="m" file="../secret.stl"/></asset></mujoco>',
    ):
        (tmp_path / "lead.xml").write_text(xml)
        with pytest.raises(ValueError):
            scenarios.validate_model_xml(str(tmp_path / "lead.xml"))
    (tmp_path / "lead.xml").write_text(f'<mujoco><include file="{sref}"/></mujoco>')
    (tmp_path / "trim.py").write_text(
        "import numpy as np\ndef make_trim(m): return lambda q: np.zeros(m.nv)\n")
    report = codesign_scorer.score_submission(str(tmp_path))
    assert report["reward"] == 0.0
    assert report["stages"]["hardware"] == 0.0


def test_trim_sandbox_isolates_memory_keys_and_subprocesses(tmp_path):
    sref = Path(codesign_oracle.CODESIGN_MODELS) / "franka" / "lead_sref.xml"
    shutil.copy(sref, tmp_path / "lead.xml")
    shutil.copytree(sref.parent / "meshes", tmp_path / "meshes")
    (tmp_path / "trim.py").write_text('''
import inspect, os, sys, threading
import numpy as np
import harness.scenarios as hs
import harness.spec as hspec

assert "PYTHONPATH" not in os.environ
assert not hasattr(hs, "compose_instance")
assert not hasattr(hspec, "HIDDEN_SEEDS")
frame = inspect.currentframe()
while frame:
    assert "jobs" not in frame.f_locals
    if "request" in frame.f_locals and isinstance(frame.f_locals["request"], dict):
        assert isinstance(frame.f_locals["request"].get("plan_probe"), list)
        assert isinstance(frame.f_locals["request"].get("ctrim"), list)
        assert "hold::129" not in repr(frame.f_locals["request"])
    frame = frame.f_back

done = []
t = threading.Thread(target=lambda: done.append(1))
t.start(); t.join()
assert done == [1]

if hasattr(os, "uname") and os.uname().sysname == "Linux":
    try:
        pid = os.fork()
    except OSError:
        pass
    else:
        if pid == 0:
            os._exit(0)
        raise AssertionError("fork succeeded inside trim sandbox")

def plan_probe(model, sample, lower, upper):
    return np.tile((lower + upper) / 2, (15, 1))

def make_trim(model):
    return lambda q: np.zeros(model.nv)
''')
    out = sandbox.run_jobs(str(tmp_path), str(tmp_path / "lead.xml"), {
        "plan_probe": {"129": {"sample": (np.zeros(7), np.zeros(7)),
                               "lower": -np.ones(7), "upper": np.ones(7)}},
        "ctrim": {"hold::129": np.zeros((2, 7))},
    })
    assert out is not None
    assert "configs" in out["plan_probe"]["129"]
    assert out["ctrim"]["hold::129"].shape == (2, 7)


@pytest.mark.skipif(os.geteuid() != 0, reason="requires root to exercise verifier privilege drop")
def test_trim_worker_drops_privileges_and_cannot_access_verifier_files(tmp_path):
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    os.chmod("/logs/verifier", 0o700)
    if Path("/tests").exists():
        os.chmod("/tests", 0o700)
    sref = Path(codesign_oracle.CODESIGN_MODELS) / "franka" / "lead_sref.xml"
    shutil.copy(sref, tmp_path / "lead.xml")
    shutil.copytree(sref.parent / "meshes", tmp_path / "meshes")
    (tmp_path / "trim.py").write_text('''
import os
import numpy as np

assert os.geteuid() == os.getegid() == 65534
assert os.getgroups() == []
for path in ("/tests", "/logs/verifier"):
    if os.path.exists(path):
        try:
            os.listdir(path)
        except PermissionError:
            pass
        else:
            raise AssertionError("private directory readable: " + path)

def make_trim(model):
    return lambda q: np.ones(model.nv) * 0.05
''')
    out = sandbox.run_jobs(str(tmp_path), str(tmp_path / "lead.xml"),
                           {"ctrim": {"hold::129": np.zeros((1, 7))}})
    assert out is not None
    np.testing.assert_allclose(out["ctrim"]["hold::129"], 0.05)
