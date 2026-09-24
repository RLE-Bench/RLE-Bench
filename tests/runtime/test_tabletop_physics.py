"""Real simulator checks for the three RGB tabletop scenes."""

from __future__ import annotations

import os
import pytest
if os.environ.get("RLEBENCH_TEST_TASK") != "task03":
    pytest.skip("family-specific physics", allow_module_level=True)

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("robosuite")

import harness.tabletop as tt  # noqa: E402
from harness.tabletop import scenes as SC  # noqa: E402
CAMS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
TOP = SC.TABLE_OFFSET[2]


def make(task, seed=7, cameras=(), **kw):
    return tt.make(task, seed=seed, camera_names=list(cameras), **kw)


def object_qpos(env):
    return {name: env.sim.data.get_joint_qpos(obj.joints[0]).copy()
            for name, obj in env.objects.items()}


# -- construction, determinism, contract ----------------------------------------

@pytest.mark.parametrize("task", ["TowerMaxHeight", "CantileverOverhang",
                                  "BalanceCoins"])
def test_scene_constructs_resets_and_scores(task):
    env = make(task)
    env.reset()
    assert env.get_ep_meta()["lang"]
    assert isinstance(env._check_success(), bool)
    assert 0.0 <= env._trial_score() <= 1.0
    env.close()


def test_same_seed_same_state():
    states = []
    for _ in range(2):
        env = make("TowerMaxHeight", seed=11)
        env.reset()
        states.append(object_qpos(env))
        env.close()
    for name in states[0]:
        np.testing.assert_allclose(states[0][name], states[1][name], atol=1e-9)


def test_action_layout_and_cameras():
    """The robot control contract: 12-D action, the three cameras, real pixels."""
    env = make("TowerMaxHeight", cameras=CAMS)
    obs = env.reset()
    low, _high = env.action_spec
    assert low.shape == (12,)
    for cam in CAMS:
        frame = obs[f"{cam}_image"]
        assert frame.ndim == 3 and frame.std() > 1.0, f"{cam} rendered black"
    # robot proprioception used by world-frame move()
    for key in ("robot0_base_pos", "robot0_base_quat",
                "robot0_base_to_eef_pos", "robot0_base_to_eef_quat",
                "robot0_gripper_qpos"):
        assert key in obs, f"missing {key}"
    env.close()


def _pan_spots(env, side_y, n=3):
    side = "left" if side_y < 0 else "right"
    center = np.array(env.position_observation()["pan_positions"][side])
    return [center + [dx, 0, env.COIN_HALF + 0.001]
            for dx in (-0.06, 0.0, 0.06)][:n]


def _load_pans(env, left, right):
    for i, spot in zip(left, _pan_spots(env, -0.20)):
        SC.teleport_obj(env, f"coin_{i}", np.array(spot))
    for i, spot in zip(right, _pan_spots(env, 0.20)):
        SC.teleport_obj(env, f"coin_{i}", np.array(spot))
    SC.settle_objects(env, 2500)


def test_balance_resolves_three_versus_three():
    env = make("BalanceCoins", seed=13)
    env.reset()
    heavy = env._heavy
    others = [i for i in range(env.N_COINS) if i != heavy]
    _load_pans(env, [heavy] + others[:2], others[2:5])
    angle = float(env.sim.data.get_joint_qpos("balance_hinge"))
    # heavy group on the -y pan: it sinks, which is a POSITIVE hinge angle
    # (axis +x) -- the sign an agent reads off the beam pose
    assert angle > 0.05, f"beam did not tip toward the heavy pan: {angle}"
    env.close()


def test_balance_stays_level_when_balanced():
    env = make("BalanceCoins", seed=13)
    env.reset()
    heavy = env._heavy
    others = [i for i in range(env.N_COINS) if i != heavy]
    _load_pans(env, others[:3], others[3:6])
    angle = float(env.sim.data.get_joint_qpos("balance_hinge"))
    assert abs(angle) < 0.03, f"beam tipped with equal loads: {angle}"
    env.close()


def test_balance_success_needs_the_heavy_coin_alone():
    env = make("BalanceCoins", seed=13)
    env.reset()
    mat = np.array([*env.MAT_POS, TOP + 0.05])
    SC.teleport_obj(env, f"coin_{env._heavy}", mat)
    SC.settle_objects(env, 400)
    assert env._check_success()
    assert env._trial_score() == 1.0     # no weighings used: still optimal
    # a second coin on the mat spoils it
    other = (env._heavy + 1) % env.N_COINS
    SC.teleport_obj(env, f"coin_{other}", mat + np.array([0.04, 0.0, 0.05]))
    SC.settle_objects(env, 400)
    assert not env._check_success()
    env.close()


# -- tower / cantilever ------------------------------------------------------------

def test_tower_stack_scores_high_and_beats_scatter():
    env = make("TowerMaxHeight", seed=7)
    env.reset()
    scattered = env._trial_score()

    # stack cubes -> plates -> cylinders (flat-stackable subset, biggest first).
    # Sampled jitter moved the dims, so each piece is dropped from just above
    # the measured top of the stack so far, then settled.
    order = ["cube_a", "plate_a", "cube_b", "cyl_b", "plate_b", "cube_c", "cyl_a"]
    z = TOP + 0.001
    for name in order:
        SC.teleport_obj(env, name, np.array([0.0, -0.25, z + 0.06]))
        SC.settle_objects(env, 700)
        m, d = env.sim.model._model, env.sim.data._data
        gids = SC._collision_gids(m, {env.obj_body_id[name]})
        z = max(SC._geom_world_top(m, d, g) for g in gids)

    stacked = env._trial_score()
    assert stacked > scattered
    assert stacked >= 0.5, f"deliberate stack scored only {stacked}"
    env.close()


@pytest.mark.parametrize("angle", [0, np.pi / 2, np.pi, 3 * np.pi / 2])
def test_cantilever_harmonic_stack_scores_high_and_floor_scores_zero(angle):
    env = make("CantileverOverhang", seed=7)
    env.reset()
    L = 2 * env.BLOCK_HALF[1]           # long axis along y, toward -y edge
    edge = env._edge_y
    f = 0.8
    shift = [0, f * L / 6, f * L / 6 + f * L / 4,
             f * L / 6 + f * L / 4 + f * L / 2]
    direction = np.array([np.sin(angle), -np.cos(angle)])
    axis = int(np.argmax(np.abs(direction)))
    edge_distance = SC.TABLE_FULL_SIZE[axis] / 2
    rotation = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
    for i in range(env.N_BLOCKS):
        xy = direction * (edge_distance - env.BLOCK_HALF[1] + f * L / 8 + shift[i])
        SC.teleport_obj(env, f"block_{i}", np.r_[
            xy, TOP + env.BLOCK_HALF[2] + i * (2 * env.BLOCK_HALF[2] + 0.001)], rotation)
    SC.settle_objects(env, 300)
    score = env._trial_score()
    assert score >= 0.4, f"harmonic stack scored only {score}"

    # blocks shoved off the table are worth nothing
    env2 = make("CantileverOverhang", seed=7)
    env2.reset()
    for i in range(env2.N_BLOCKS):
        SC.teleport_obj(env2, f"block_{i}", np.array([-0.10, edge - 0.4, 0.2]))
    assert env2._trial_score() == 0.0
    env2.close()
    env.close()


def test_balance_fixtures_visible_in_default_camera():
    env = make("BalanceCoins", cameras=CAMS, camera_height=256, camera_width=256)
    try:
        obs = env.reset()
        model = env.sim.model
        visible = env.sim._render_context_offscreen.vopt.geomgroup
        for name in model.geom_names:
            if name == "answer_mat" or (name.startswith("balance_") and name.endswith("_visual")):
                assert visible[model.geom_group[model.geom_name2id(name)]], name
        import mujoco
        import xml.etree.ElementTree as ET
        xml = ET.fromstring(env.model.get_xml())
        for parent in xml.iter():
            for geom in list(parent.findall("geom")):
                name = geom.get("name", "")
                if name.startswith("balance_") and name.endswith("_visual"):
                    parent.remove(geom)
        original = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        np.testing.assert_array_equal(model.body_mass, original.body_mass)
        np.testing.assert_array_equal(model.body_inertia, original.body_inertia)
        rgb = obs["robot0_agentview_left_image"].astype(float)
        green = (rgb[..., 1] > 1.4 * rgb[..., 0]) & (rgb[..., 1] > 1.4 * rgb[..., 2])
        assert green.sum() > 20, "green answer mat missing from agent camera"
    finally:
        env.close()


def test_balance_positions_follow_physics_without_revealing_weights():
    env = make("BalanceCoins", seed=13)
    try:
        env.reset()
        before = env.position_observation()
        assert set(before) == {"cube_positions", "pan_positions"}
        assert set(before["cube_positions"]) == {f"coin_{i}" for i in range(9)}
        assert set(before["pan_positions"]) == {"left", "right"}
        for side, gid in env._pan_geom.items():
            assert env.sim.model.geom_size[gid, 0] == pytest.approx(0.1125)
        left, right = (np.array(before["pan_positions"][key]) for key in ("left", "right"))
        assert np.linalg.norm(left - right) == pytest.approx(0.5)
        heavy = env._heavy
        others = [i for i in range(env.N_COINS) if i != heavy]
        _load_pans(env, [heavy] + others[:2], others[2:5])
        after = env.position_observation()
        assert after["pan_positions"]["left"][2] < after["pan_positions"]["right"][2]
        for name, position in after["cube_positions"].items():
            np.testing.assert_allclose(position, SC.obj_pos(env, name))
    finally:
        env.close()
