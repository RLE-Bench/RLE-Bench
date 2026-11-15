"""Development and verification share the submitted-package boundary."""
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from harness import dev_runner, runtime, sandbox, scorer, spec


GOOD = '''
from helper import control
import mujoco
class Policy:
    def reset(self, cell, seed):
        assert mujoco.MjModel.from_binary_path(cell["model_file"]).nu == 7
        assert "damage_force_n" in cell
        self.ctrl = control(cell)
    def act(self, obs):
        return self.ctrl
def make_policy():
    return Policy()
'''


@pytest.fixture
def package(tmp_path):
    path = tmp_path / "policy"
    path.mkdir(mode=0o700)
    (path / "policy.py").write_text(GOOD)
    (path / "helper.py").write_text('def control(cell): return cell["arm_home"] + [0.]\n')
    (path / "helper.py").chmod(0o600)
    return path


@pytest.mark.parametrize("code", ["from harness import spec", "import missing_custom_module",
                                  "def make_policy(): raise ValueError('factory failed')"])
def test_dev_rejects_unloadable_policy_before_simulation(package, monkeypatch, capsys, code):
    (package / "policy.py").write_text(code)
    monkeypatch.setenv("PYTHONPATH", str(Path(dev_runner.__file__).parents[1]))
    monkeypatch.setattr(sys, "argv", ["dev_runner.py", str(package)])
    monkeypatch.setattr(runtime, "run_episode", lambda *a, **kw: pytest.fail("simulation started"))
    assert dev_runner.main() == 1
    assert "policy load failed" in capsys.readouterr().err


def test_dev_runs_bundled_policy_and_preserves_source(package, monkeypatch, capsys):
    run_episode = runtime.run_episode

    def short_run(policy, seed, **kwargs):
        assert isinstance(policy, sandbox.PolicyProcess)
        assert kwargs["wall_budget_s"] == spec.EPISODE_WALL_BUDGET_S
        kwargs["render"] = False
        return run_episode(policy, seed, **kwargs)

    monkeypatch.setattr(runtime, "run_episode", short_run)
    monkeypatch.setattr(sys, "argv", ["dev_runner.py", str(package), "--budget", "0.1"])
    assert dev_runner.main() == 0
    assert '"policy_faults": 0' in capsys.readouterr().out
    assert package.stat().st_mode & 0o777 == 0o700
    assert (package / "helper.py").stat().st_mode & 0o777 == 0o600


def test_verifier_load_failure_skips_all_scenarios(package, monkeypatch):
    (package / "policy.py").write_text("from harness import spec")
    monkeypatch.setattr(runtime, "run_episode", lambda *a, **kw: pytest.fail("simulation started"))
    report = scorer.score_submission(str(package), seeds={"smoke": 35, "eval": [16, 307]})
    assert report["reward"] == 0 and report["gated"]
    assert report["gate"]["load"] is False
    assert report["episodes"] == []
    assert "ModuleNotFoundError" in report["errors"][0]


RESET_FAILS = GOOD.replace("assert mujoco", "raise ValueError('reset failed')\n        assert mujoco")


def test_dev_stops_when_reset_fails(package, monkeypatch, capsys):
    (package / "policy.py").write_text(RESET_FAILS)
    monkeypatch.setattr(sys, "argv", ["dev_runner.py", str(package)])
    assert dev_runner.main() == 1
    assert "ValueError: reset failed" in capsys.readouterr().err


def test_verifier_reset_failure_skips_remaining_scenarios(package, monkeypatch):
    (package / "policy.py").write_text(RESET_FAILS)
    calls = []
    run = scorer._run_sandboxed
    monkeypatch.setattr(scorer, "_run_sandboxed", lambda *a, **kw: calls.append(1) or run(*a, **kw))
    report = scorer.score_submission(str(package), seeds={"smoke": 35, "eval": [16, 307]})
    assert report["reward"] == 0 and report["gated"] and len(calls) == 1
    assert report["episodes"] == [] and "reset failed" in report["errors"][0]


@pytest.mark.skipif(os.geteuid() != 0, reason="requires root to exercise verifier privilege drop")
def test_policy_cannot_read_private_code_or_seeds(package, tmp_path):
    private = tmp_path / "tests"
    private.mkdir(mode=0o700)
    # Make the ancestor traversable, so denial is due to the verifier boundary.
    for parent in private.parents:
        if parent == Path("/tmp"):
            break
        parent.chmod(0o755)
    paths = [private / "harness" / name for name in ("scorer.py", "eval_seeds.json")]
    paths[0].parent.mkdir()
    for path in paths:
        path.write_text("private sentinel")
    # In the verifier image also probe its actual hidden files.
    paths += [p for p in (Path("/tests/harness/scorer.py"),
                          Path("/tests/harness/eval_seeds.json")) if p.exists()]
    probe = f'''
import os
assert os.geteuid() == os.getegid() == 65534
assert os.getgroups() == []
assert "PYTHONPATH" not in os.environ
for path in {list(map(str, paths))!r}:
    try:
        open(path).read()
    except PermissionError:
        pass
    else:
        raise AssertionError("private file readable: " + path)
'''
    (package / "policy.py").write_text(probe + GOOD)
    with sandbox.PolicyProcess(str(package)) as policy:
        policy.check_loaded()
        cell = spec.cell_spec()
        cell["damage_force_n"] = 1.0
        policy.reset(cell, 0)
        np.testing.assert_array_equal(policy.act({}), cell["arm_home"] + [0.])
        assert policy.dead is None, policy.stderr_tail()


def test_package_symlinks_cannot_copy_or_chmod_private_files(package, tmp_path):
    secret = tmp_path / "private_seed"
    secret.write_text("private sentinel")
    secret.chmod(0o600)
    (package / "linked_seed").symlink_to(secret)
    with sandbox.PolicyProcess(str(package)) as policy:
        with pytest.raises(sandbox.PolicyLoadError, match="symbolic links"):
            policy.check_loaded()
        assert (Path(policy.work) / "policy" / "linked_seed").is_symlink()
    assert secret.stat().st_mode & 0o777 == 0o600


def test_verifier_staging_preserves_links_for_rejection(package, tmp_path, monkeypatch):
    from harness import score_task

    private = tmp_path / "private_seed"
    private.write_text("private sentinel")
    (package / "linked_seed").symlink_to(private)
    staging = tmp_path / "staged"
    monkeypatch.setattr(score_task, "ARTIFACTS", str(package.parent))
    monkeypatch.setattr(score_task, "STAGING", str(staging))
    monkeypatch.setattr(score_task, "_media", lambda: None)
    reports = []
    monkeypatch.setattr(score_task, "_write", lambda reward, report=None: reports.append(report))
    score_task.main()
    assert staging.joinpath("linked_seed").is_symlink()
    assert staging.stat().st_mode & 0o077 == 0
    assert reports[0]["gate"]["load"] is False
    assert "symbolic links" in reports[0]["errors"][0]


def test_verifier_all_faulting_smoke_skips_remaining_scenarios(package, monkeypatch):
    (package / "policy.py").write_text(
        GOOD.replace("return self.ctrl", "raise RuntimeError('every tick fails')"))
    monkeypatch.setattr(scorer.config, "SMOKE_T", 0.2)
    calls = []
    run = scorer._run_sandboxed
    monkeypatch.setattr(scorer, "_run_sandboxed",
                        lambda *a, **kw: calls.append(1) or run(*a, **kw))
    report = scorer.score_submission(str(package), seeds={"smoke": 35, "eval": [16, 307]})
    assert report["reward"] == 0.0 and report["gated"] and len(calls) == 1
    assert report["gate"]["contract"] is False
    assert report["episodes"] == []
    assert "every tick fails" in report["errors"][0]


def test_sandbox_allows_threads_and_blocks_subprocesses(package):
    (package / "memo.txt").write_text("seedless")
    (package / "memo.txt").chmod(0o666)
    probe = '''
import os, threading
done = []
t = threading.Thread(target=lambda: done.append(True))
t.start(); t.join()
assert done == [True]
if hasattr(os, "uname") and os.uname().sysname == "Linux":
    try:
        pid = os.fork()
    except OSError:
        pass
    else:
        if pid == 0:
            os._exit(0)
        raise AssertionError("fork succeeded inside policy sandbox")
'''
    (package / "policy.py").write_text(probe + GOOD)
    with sandbox.PolicyProcess(str(package)) as policy:
        assert (Path(policy.work) / "policy" / "memo.txt").stat().st_mode & 0o022 == 0
        policy.check_loaded()
        cell = spec.cell_spec()
        cell["damage_force_n"] = 1.0
        policy.reset(cell, 0)
        np.testing.assert_array_equal(policy.act({}), cell["arm_home"] + [0.])
        assert policy.dead is None, policy.stderr_tail()
