"""Task07 runtime + sandbox tests: the closed control loop, the latch
path, fault containment, and the interactive sandbox contract."""
import os
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np


from harness import (config, episodes, runtime, sandbox, scene,
                                scorer, spec)

HOME_CTRL = np.array([*spec.ARM_HOME, 0.0])
SEED = spec.DESIGN_SEEDS[0]


def _short(policy, budget_t=3.0, seed=SEED, **kw):
    return runtime.run_episode(policy, seed, budget_t=budget_t,
                               render=False, **kw)


# ---------------------------------------------------------------------------
# In-process control loop
# ---------------------------------------------------------------------------
def test_null_policy_episode_and_determinism():
    log1 = _short(runtime.NullPolicy())
    log2 = _short(runtime.NullPolicy())
    assert log1.n_parts >= spec.N_PARTS_RANGE[0] - 4
    assert log1.clear_times == {} and log1.floor_drops == 0
    assert log1.policy_faults == 0 and log1.aborted is None
    assert log1.digest == log2.digest          # invariant #3, full loop
    assert log1.ticks == int(3.0 * spec.CTRL_HZ)
    assert log1.metrics()["penalty_events"] == []


def test_penalty_event_diagnostics(monkeypatch):
    """Penalty latches retain the state and contact that caused them."""
    real_positions = scene.part_positions
    real_force_details = scene.part_contact_force_details
    calls = {"positions": 0, "forces": 0}

    def positions_with_one_drop(data, n):
        values = real_positions(data, n)
        calls["positions"] += 1
        if calls["positions"] == 1:
            values[0, 2] = spec.FLOOR_Z - 0.001
        return values

    def forces_with_one_damage(model, data, n):
        values, contacts = real_force_details(model, data, n)
        calls["forces"] += 1
        if calls["forces"] <= config.DMG_TICKS:
            values[0] = config.DAMAGE_FORCE_N + 100.0
        return values, contacts

    monkeypatch.setattr(scene, "part_positions", positions_with_one_drop)
    monkeypatch.setattr(scene, "part_contact_force_details",
                        forces_with_one_damage)
    log = _short(runtime.NullPolicy(), budget_t=0.15)
    events = log.metrics()["penalty_events"]
    floor_event = next(e for e in events if e["type"] == "floor_drop")
    damage_event = next(e for e in events if e["type"] == "damage")

    assert floor_event["part_id"] == 0
    assert floor_event["part_body"] == "part0"
    assert len(floor_event["part_pos_world_m"]) == 3
    assert floor_event["floor_z_threshold_m"] == spec.FLOOR_Z
    assert floor_event["last_release"] is None
    assert isinstance(floor_event["current_contacts"], list)
    assert len(floor_event["linear_velocity_world_mps"]) == 3
    assert len(floor_event["angular_velocity_world_radps"]) == 3

    assert damage_event["part_id"] == 0
    assert damage_event["force_n"] == config.DAMAGE_FORCE_N + 100.0
    assert damage_event["previous_force_n"] == damage_event["force_n"]
    assert damage_event["sustained_force_n"] > config.DAMAGE_FORCE_N
    assert damage_event["threshold_n"] == config.DAMAGE_FORCE_N
    assert damage_event["streak_ticks"] == config.DMG_TICKS
    assert "contact" in damage_event


def test_real_contact_detail_names_bodies_and_force():
    model, data = episodes.build_episode_scene(SEED)
    scene.settle(model, data, 0.05)
    n = scene.n_parts(model)
    forces, indices = scene.part_contact_force_details(model, data, n)
    i = int(np.argmax(forces))
    detail = scene.part_contact_detail(model, data, i, int(indices[i]))
    linear, angular = scene.part_velocity(model, data, i)

    assert forces[i] > 0.0
    assert detail is not None
    assert detail["part_geom"].startswith(f"part{i}_")
    assert detail["other_geom"]
    assert detail["other_body"]
    assert detail["category"].startswith("part_")
    assert len(detail["pos_world_m"]) == 3
    assert len(detail["normal_world"]) == 3
    assert detail["force_n"] == forces[i]
    assert linear.shape == (3,) and angular.shape == (3,)


def test_policy_seed_is_independent_from_world_seed():
    class SeedProbe(runtime.NullPolicy):
        def reset(self, cell, seed):
            self.seed = seed

    policy = SeedProbe()
    expected = scorer._policy_seed(SEED)
    _short(policy, budget_t=0.0, policy_seed=expected)
    assert policy.seed == expected
    assert policy.seed != SEED
    assert scorer._policy_seed(SEED) == expected
    assert scorer._policy_seed(SEED + 1) != expected

    defaulted = SeedProbe()
    _short(defaulted, budget_t=0.0)
    assert defaulted.seed == 0


def test_policy_wall_budget_uses_parent_side_policy_meter_only():
    class Metered(runtime.NullPolicy):
        def __init__(self):
            self.policy_wall_s = 0.0

        def act(self, obs):
            self.policy_wall_s += 0.06
            return HOME_CTRL

    log = _short(Metered(), budget_t=1.0, wall_budget_s=0.10)
    assert log.aborted == "wall_budget"
    assert log.ticks == 1       # the action that crosses the cap is not applied


def test_policy_wall_budget_ignores_runtime_monotonic_clock(monkeypatch):
    class Metered(runtime.NullPolicy):
        policy_wall_s = 0.0

    def forbidden_clock():
        raise AssertionError("harness wall clock must not define policy budget")

    monkeypatch.setattr(time, "monotonic", forbidden_clock)
    log = _short(Metered(), budget_t=0.2, wall_budget_s=0.001)
    assert log.aborted is None
    assert log.ticks == int(0.2 * spec.CTRL_HZ)


def test_faulting_policy_is_contained():
    class Raiser:
        def reset(self, cell, seed):
            pass

        def act(self, obs):
            raise RuntimeError("boom")

    log = _short(Raiser(), budget_t=1.0)
    assert log.policy_faults == log.ticks
    assert log.aborted is None                 # physics ran to budget
    assert log.fault_reasons and "boom" in log.fault_reasons[0]
    reported = log.metrics(1.0)
    assert reported["ticks"] == log.ticks
    assert reported["fault_reasons"] == log.fault_reasons


def test_bad_ctrl_shape_is_contained():
    class Bad:
        def reset(self, cell, seed):
            pass

        def act(self, obs):
            return np.zeros(3)

    log = _short(Bad(), budget_t=1.0)
    assert log.policy_faults == log.ticks


def test_nonfinite_ctrl_is_contained():
    class NaNs:
        def reset(self, cell, seed):
            pass

        def act(self, obs):
            return np.full(spec.N_CTRL, np.nan)

    log = _short(NaNs(), budget_t=1.0)
    assert log.policy_faults == log.ticks
    assert log.aborted is None


def test_obs_contract_keys():
    seen = {}

    class Probe(runtime.NullPolicy):
        def act(self, obs):
            seen.setdefault("keys", set()).update(obs.keys())
            seen.setdefault("n", 0)
            seen["n"] = seen["n"] + 1
            if "overhead_rgb" in obs:
                seen["frames"] = seen.get("frames", 0) + 1
            return HOME_CTRL

    runtime.run_episode(Probe(), SEED, budget_t=1.0, render=True)
    assert {"qpos", "qvel", "tau", "ft", "mag_on", "t"} <= seen["keys"]
    # no part-present channel: a grab is only observable through `ft`
    assert "held" not in seen["keys"]
    assert {"overhead_rgb", "overhead_depth", "wrist_rgb", "wrist_depth",
            "wrist_T_world_cam"} <= seen["keys"]
    # frames at FRAME_HZ, control at CTRL_HZ
    assert seen["frames"] == seen["n"] // spec.FRAME_EVERY_TICKS


def test_clear_latch_through_runtime(monkeypatch):
    """A part resting in the drop zone latches, is graveyarded, and ends
    the episode when it was the only part."""
    zc = (np.asarray(spec.ZONE_LO) + np.asarray(spec.ZONE_HI)) / 2
    start = np.array([zc[0], zc[1], spec.BELT_TOP + 0.02])

    def one_part_scene(seed, with_arm=True):
        return scene.build_scene(
            part_poses=[(start, np.array([1.0, 0, 0, 0]))], seed=seed)

    monkeypatch.setattr(episodes, "build_episode_scene", one_part_scene)
    log = runtime.run_episode(runtime.NullPolicy(), SEED, budget_t=10.0,
                              render=False)
    assert log.clear_times.keys() == {0}
    t = log.clear_times[0]
    assert spec.CLEAR_REST_S <= t < 3.0
    assert log.ticks < int(10.0 * spec.CTRL_HZ)      # early exit
    assert log.floor_drops == 0                      # graveyard excluded
    m = log.metrics()
    assert m["clear_frac"] == 1.0 and m["parts_cleared"] == 1
    assert np.allclose(log.final_positions[0][:2],
                       [spec.GRAVEYARD[0], spec.GRAVEYARD[1]])


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------
GOOD_POLICY = '''
import mujoco
import numpy as np

class P:
    def reset(self, cell, seed):
        model = mujoco.MjModel.from_binary_path(cell["model_file"])
        assert model.nu == 7 and cell["n_ctrl"] == 8
        self.ctrl = np.array(cell["arm_home"] + [0.0])
        print("agent print must not corrupt the protocol")

    def act(self, obs):
        assert obs["qpos"].shape == (7,)
        assert obs["mag_on"] in (0.0, 1.0)
        return self.ctrl

def make_policy():
    return P()
'''


def _write_submission(tmp_path, code):
    d = tmp_path / "policy"
    d.mkdir()
    (d / "policy.py").write_text(code)
    return str(d)


def test_sandboxed_policy_matches_inprocess(tmp_path):
    sub = _write_submission(tmp_path, GOOD_POLICY)
    ref = _short(runtime.NullPolicy(), budget_t=2.0)
    with sandbox.PolicyProcess(sub) as p:
        log = _short(p, budget_t=2.0)
        assert p.dead is None, p.stderr_tail()
        assert p.reset_wall_s > 0.0 and p.act_wall_s > 0.0
        assert p.policy_wall_s == p.reset_wall_s + p.act_wall_s
    assert log.policy_faults == 0
    assert log.digest == ref.digest    # same ctrl stream => same physics


def test_scorer_exposes_sandbox_timing_diagnostics(tmp_path):
    sub = _write_submission(tmp_path, GOOD_POLICY)
    log, diagnostics = scorer._run_sandboxed(
        sub, SEED, budget_t=0.1, wall_budget_s=1.0)
    assert log.policy_faults == 0
    assert diagnostics["n_acts"] == log.ticks
    assert diagnostics["reset_wall_s"] > 0.0
    assert diagnostics["act_wall_s"] > 0.0
    assert diagnostics["policy_wall_s"] > 0.0
    assert diagnostics["episode_wall_s"] >= diagnostics["policy_wall_s"]
    assert diagnostics["harness_wall_s"] >= 0.0
    assert diagnostics["sandbox_dead"] is None
    assert "agent print" in diagnostics["sandbox_stderr_tail"]


def test_sandbox_missing_policy(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with sandbox.PolicyProcess(str(d)) as p:
        assert p.dead == "policy.py missing"
        log = _short(p, budget_t=1.0)
    assert log.aborted == "policy_startup"
    assert log.ticks == 0 and log.policy_faults == 1


def test_sandbox_hang_is_killed(tmp_path):
    sub = _write_submission(tmp_path, '''
import time
import numpy as np

class P:
    def reset(self, cell, seed):
        pass

    def act(self, obs):
        time.sleep(60)

def make_policy():
    return P()
''')
    with sandbox.PolicyProcess(sub, act_timeout_s=1.0) as p:
        log = _short(p, budget_t=1.0)
        assert p.dead is not None and "timeout" in p.dead
    assert log.policy_faults == log.ticks
    assert log.aborted is None


def test_sandbox_child_death_is_contained(tmp_path):
    sub = _write_submission(tmp_path, '''
import os
import numpy as np

class P:
    def __init__(self):
        self.n = 0

    def reset(self, cell, seed):
        pass

    def act(self, obs):
        self.n += 1
        if self.n > 3:
            os._exit(1)
        return np.array(cell_home)

cell_home = None

def make_policy():
    global cell_home
    cell_home = [0.0] * 7 + [0.0]
    return P()
''')
    with sandbox.PolicyProcess(sub) as p:
        log = _short(p, budget_t=1.0)
        assert p.dead is not None
    assert 0 < log.policy_faults >= log.ticks - 3
    assert log.aborted is None


def test_sandbox_exceptions_reported_not_fatal(tmp_path):
    sub = _write_submission(tmp_path, '''
import numpy as np

class P:
    def __init__(self):
        self.n = 0

    def reset(self, cell, seed):
        pass

    def act(self, obs):
        self.n += 1
        if self.n % 2 == 0:
            raise ValueError("flaky agent")
        return np.array([0.0] * 7 + [255.0])

def make_policy():
    return P()
''')
    with sandbox.PolicyProcess(sub) as p:
        log = _short(p, budget_t=1.0)
        assert p.dead is None            # exceptions never kill the child
        assert "flaky agent" in p.stderr_tail()
    assert 0 < log.policy_faults < log.ticks


def test_tool_bin_contact_is_measured(monkeypatch):
    """Ramming the tool into a bin wall registers tool-bin force (the
    bin-hit penalty's sensor), and an untouched run registers none."""
    from harness import golden

    def empty_bin(seed, with_arm=True):
        return scene.build_scene(part_poses=[], seed=seed)

    monkeypatch.setattr(episodes, "build_episode_scene", empty_bin)

    class Rammer:
        privileged = True

        def bind(self, model, data):
            gp = golden.GoldenPicker()
            gp.bind(model, data)
            # a point INSIDE the -y wall material, low in the bin: the
            # position actuators press the probe into the wall forever
            target = [spec.BIN_POS[0], spec.BIN_LO[1] - 0.02,
                      spec.STAND_H + 0.08]
            self.q, _ = gp._ik(target, gp._face_R(), gp._q_now())

        def reset(self, cell, seed):
            pass

        def act(self, obs):
            return np.array([*self.q, 0.0])

    log = runtime.run_episode(Rammer(), SEED, budget_t=4.0, render=False)
    assert log.peak_tool_bin > 50.0

    calm = runtime.run_episode(runtime.NullPolicy(), SEED, budget_t=2.0,
                               render=False)
    assert calm.peak_tool_bin == 0.0 and calm.bin_hits == 0


def test_video_recording_does_not_touch_physics(tmp_path):
    """The verifier records eval episodes; rendering must be a pure
    observer — identical digest with and without the recorder."""
    import shutil as _shutil
    if _shutil.which("ffmpeg") is None:
        import pytest
        pytest.skip("ffmpeg unavailable")
    plain = _short(runtime.NullPolicy(), budget_t=2.0)
    from rlebench.core.media import Media
    media = Media(tmp_path)
    writer = media.video("ep.mp4")
    taped = runtime.run_episode(runtime.NullPolicy(), SEED, budget_t=2.0,
                                render=False, video=writer,
                                video_mode="camera", video_every=2)
    media.finish(writer)
    assert taped.digest == plain.digest
    assert media.close()["files"] == ["ep.mp4"]
    assert os.path.getsize(tmp_path / "ep.mp4") > 10000
