import os
import pytest
if os.environ.get("RLEBENCH_TEST_TASK") != "task01":
    pytest.skip("family-specific checks", allow_module_level=True)

import numpy as np
from harness.skills import camera as C, geometry as G, transforms as T

def test_quat_and_mat_round_trip():
    for quat in ([0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0],
                 [0.5, 0.5, 0.5, 0.5], [0.0, 0.3826834, 0.0, 0.9238795]):
        mat = T.quat_to_mat(quat)
        np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(T.quat_to_mat(T.mat_to_quat(mat)), mat, atol=1e-9)


def test_quat_convention_is_xyzw():
    """xyzw, like every observation key. Read as wxyz these are different rotations."""
    # A quarter turn about z: xyzw (0, 0, sin45, cos45) takes +x to +y.
    quat = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    np.testing.assert_allclose(T.quat_to_mat(quat) @ [1.0, 0, 0], [0, 1, 0], atol=1e-9)


def test_invert_transform_is_the_inverse():
    mat = T.make_transform([1.0, -2.0, 0.5], [0.0, 0.0, np.sin(0.3), np.cos(0.3)])
    np.testing.assert_allclose(mat @ T.invert_transform(mat), np.eye(4), atol=1e-9)


def test_decompose_transform_recovers_what_make_transform_built():
    pos, quat = [0.1, 0.2, 0.3], [0.0, np.sin(0.2), 0.0, np.cos(0.2)]
    back_pos, back_quat = T.decompose_transform(T.make_transform(pos, quat))
    np.testing.assert_allclose(back_pos, pos, atol=1e-9)
    np.testing.assert_allclose(T.quat_to_mat(back_quat), T.quat_to_mat(quat), atol=1e-9)


def test_transform_points_takes_one_point_or_many():
    mat = T.make_transform([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(T.transform_points(mat, [0.0, 0.0, 0.0]), [1, 0, 0])
    out = T.transform_points(mat, [[0.0, 0, 0], [0.0, 1, 0]])
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, [[1, 0, 0], [1, 1, 0]])


def test_world_to_base_undoes_a_yawed_base():
    """The gotcha the whole task warns about, in one assertion.

    A base yawed 90 degrees and standing at (2, 1, 0): a world point one metre in front
    of it along +y is one metre along the base's own +x.
    """
    obs = {"robot0_base_pos": [2.0, 1.0, 0.0],
           "robot0_base_quat": [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]}
    np.testing.assert_allclose(T.world_to_base(obs, [2.0, 2.0, 0.0]), [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(
        T.base_to_world(obs, T.world_to_base(obs, [0.3, -0.4, 0.9])),
        [0.3, -0.4, 0.9], atol=1e-9)


def test_mat_to_axisangle_recovers_axis_times_angle():
    np.testing.assert_allclose(
        T.mat_to_axisangle(T.quat_to_mat([0.0, 0.0, np.sin(0.3), np.cos(0.3)])),
        [0, 0, 0.6], atol=1e-9)
    np.testing.assert_allclose(T.mat_to_axisangle(np.eye(3)), [0, 0, 0])
    # A half turn, where the trace formula degenerates.
    np.testing.assert_allclose(
        T.mat_to_axisangle(np.diag([-1.0, -1.0, 1.0])), [0, 0, np.pi], atol=1e-9)


def test_mat_to_axisangle_takes_the_short_way_round():
    """A 5.8 rad turn about z is commanded as -0.48 rad, not as most of a full circle."""
    big = T.quat_to_mat([0.0, 0.0, np.sin(2.9), np.cos(2.9)])
    np.testing.assert_allclose(
        T.mat_to_axisangle(big), [0, 0, 5.8 - 2 * np.pi], atol=1e-9)


def test_normalize_leaves_a_zero_vector_alone():
    np.testing.assert_allclose(T.normalize([0.0, 0.0, 0.0]), [0, 0, 0])
    np.testing.assert_allclose(np.linalg.norm(T.normalize([3.0, 4.0, 0.0])), 1.0)


def test_intrinsics_match_the_published_fovy():
    """The instruction publishes f = (H/2) / tan(fovy/2); this must be that number."""
    k = C.intrinsics("robot0_agentview_left", 128)
    assert k[0, 0] == pytest.approx((128 / 2) / np.tan(np.radians(60) / 2))
    assert C.intrinsics("robot0_eye_in_hand", 128)[0, 0] == pytest.approx(
        (128 / 2) / np.tan(np.radians(75) / 2))


def test_focal_length_scales_with_the_size_you_rendered():
    small = C.intrinsics("robot0_agentview_left", 128)[0, 0]
    big = C.intrinsics("robot0_agentview_left", 512)[0, 0]
    assert big == pytest.approx(4 * small)


def test_intrinsics_accepts_an_observation_key_name():
    a = C.intrinsics("robot0_eye_in_hand_image", 256)
    b = C.intrinsics("robot0_eye_in_hand_depth", 256)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(a, C.intrinsics("robot0_eye_in_hand", 256))


def test_unknown_camera_is_refused():
    with pytest.raises(KeyError):
        C.intrinsics("robot0_frontview", 128)


def test_deproject_inverts_a_known_projection():
    """Project a point by hand, then deproject the pixel it landed on."""
    k = C.intrinsics("robot0_agentview_left", 64)
    fx, cx, cy = k[0, 0], k[0, 2], k[1, 2]
    point = np.array([0.05, -0.03, 0.8])
    u = int(round(point[0] * fx / point[2] + cx))
    v = int(round(point[1] * fx / point[2] + cy))
    depth = np.full((64, 64), np.nan)
    depth[v, u] = point[2]
    np.testing.assert_allclose(C.deproject(depth, k)[v, u], point, atol=2e-2)


def test_deproject_marks_unusable_depth_rather_than_placing_it_at_the_origin():
    k = C.intrinsics("robot0_agentview_left", 8)
    depth = np.zeros((8, 8))
    depth[0, 0] = np.inf
    assert np.isnan(C.deproject(depth, k)).all()


def test_remove_support_plane_takes_the_counter_and_leaves_the_object():
    counter = _slab(0.90)
    obj = np.random.default_rng(0).uniform([-0.02, -0.02, 0.95],
                                           [0.02, 0.02, 1.00], size=(200, 3))
    kept = G.remove_support_plane(np.vstack([counter, obj]))
    assert len(kept) == len(obj)
    assert kept[:, 2].min() > 0.92


def test_remove_support_plane_leaves_a_cloud_with_no_surface_alone():
    """No band holds enough of the cloud, so nothing is a surface and nothing goes."""
    blob = np.random.default_rng(1).uniform(-0.5, 0.5, size=(500, 3))
    assert len(G.remove_support_plane(blob)) == len(blob)


def test_cluster_points_separates_two_objects():
    rng = np.random.default_rng(2)
    a = rng.normal([0.0, 0.0, 1.0], 0.01, size=(120, 3))
    b = rng.normal([0.5, 0.0, 1.0], 0.01, size=(60, 3))
    groups = G.cluster_points(np.vstack([a, b]), radius=0.03, min_size=20)
    assert len(groups) == 2
    assert len(groups[0]) > len(groups[1])          # largest first
    np.testing.assert_allclose(groups[0].mean(axis=0), [0, 0, 1], atol=0.02)


def test_cluster_points_drops_specks_and_survives_an_empty_cloud():
    assert G.cluster_points(np.zeros((0, 3))) == []
    speck = np.random.default_rng(3).normal(0, 0.001, size=(5, 3))
    assert G.cluster_points(speck, radius=0.03, min_size=20) == []


def test_oriented_bbox_recovers_a_rotated_box():
    rng = np.random.default_rng(4)
    half = np.array([0.10, 0.04, 0.03])
    local = rng.uniform(-half, half, size=(4000, 3))
    yaw = 0.6
    rot = T.quat_to_mat([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
    cloud = local @ rot.T + np.array([1.0, 2.0, 0.9])

    box = G.oriented_bbox(cloud)
    np.testing.assert_allclose(box["center"], [1.0, 2.0, 0.9], atol=0.01)
    np.testing.assert_allclose(np.sort(box["extent"])[::-1], np.sort(2 * half)[::-1],
                               atol=0.02)
    # The box's major axis is the box's long side, up to sign.
    axes = T.quat_to_mat(box["quat"])
    assert abs(float(axes[:, 0] @ (rot @ [1.0, 0, 0]))) > 0.99


def test_oriented_bbox_is_gravity_aligned():
    """Its z axis is up whatever the cloud looks like -- objects sit on surfaces."""
    tilted = np.random.default_rng(5).uniform(-1, 1, size=(500, 3)) * [1.0, 0.2, 0.6]
    axes = T.quat_to_mat(G.oriented_bbox(tilted)["quat"])
    np.testing.assert_allclose(axes[:, 2], [0, 0, 1], atol=1e-9)


def test_oriented_bbox_needs_a_point():
    with pytest.raises(ValueError):
        G.oriented_bbox(np.zeros((0, 3)))


def test_select_top_down_grasp_prefers_the_downward_one():
    grasps = np.stack([_grasp([1.0, 0.0, 0.0]), _grasp([0.0, 0.0, -1.0]),
                       _grasp([0.3, 0.0, -1.0])])
    assert G.select_top_down_grasp(grasps) == 1


def test_select_top_down_grasp_breaks_ties_on_score():
    grasps = np.stack([_grasp([0.0, 0.0, -1.0]), _grasp([0.0, 0.0, -1.0])])
    assert G.select_top_down_grasp(grasps, scores=[0.1, 0.9]) == 1


def test_select_top_down_grasp_returns_none_when_nothing_is_top_down():
    """A real answer: approaching from above will not work here."""
    grasps = np.stack([_grasp([1.0, 0.0, 0.0]), _grasp([0.0, 1.0, 0.0])])
    assert G.select_top_down_grasp(grasps, max_tilt_deg=45.0) is None
    assert G.select_top_down_grasp(grasps, max_tilt_deg=None) is not None


def test_select_top_down_grasp_refuses_a_wrong_shape():
    with pytest.raises(ValueError):
        G.select_top_down_grasp(np.zeros((3, 3)))
    assert G.select_top_down_grasp(np.zeros((0, 4, 4))) is None


def test_agentview_extrinsics_are_proper_rotations():
    for cam in C._AGENTVIEW_POSE:
        rot, off = C.camera_pose_in_base(cam)
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-9)
        assert off.shape == (3,)


def test_the_two_agentviews_are_mirrored_across_the_robot():
    left, right = (C.camera_pose_in_base(c)[1] for c in C._AGENTVIEW_POSE)
    np.testing.assert_allclose(left, [-0.5, 0.35, 1.05])
    np.testing.assert_allclose(right, [-0.5, -0.35, 1.05])


def test_the_wrist_extrinsic_follows_the_hand():
    """It moves with the end-effector, so the same camera has a different base pose for
    a different arm configuration -- and the offset in the HAND frame stays put."""
    from harness.skills import transforms as TR

    a = {"robot0_base_to_eef_pos": [0.4, 0.0, 1.0],
         "robot0_base_to_eef_quat": [0.0, 0.0, 0.0, 1.0]}
    b = {"robot0_base_to_eef_pos": [0.4, 0.0, 1.0],
         "robot0_base_to_eef_quat": [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]}
    rot_a, off_a = C.camera_pose_in_base(C.WRIST, a)
    rot_b, off_b = C.camera_pose_in_base(C.WRIST, b)
    assert not np.allclose(rot_a, rot_b)

    # Back out the hand-frame offset from each: it is the same constant both times.
    for obs, rot, off in ((a, rot_a, off_a), (b, rot_b, off_b)):
        hand = TR.quat_to_mat(obs["robot0_base_to_eef_quat"])
        np.testing.assert_allclose(
            hand.T @ (off - np.asarray(obs["robot0_base_to_eef_pos"], dtype=float)),
            [0.05, 0.0, -0.097], atol=1e-9)


def test_to_base_is_the_inverse_of_the_camera_pose():
    rot, off = C.camera_pose_in_base("robot0_agentview_left")
    pts = np.random.default_rng(0).normal(size=(20, 3))
    back = (C.to_base(pts, "robot0_agentview_left") - off) @ rot
    np.testing.assert_allclose(back, pts, atol=1e-9)


def test_pixel_to_base_point_is_none_where_there_is_no_depth():
    obs = {"robot0_agentview_left_depth": np.zeros((16, 16)),
           "robot0_agentview_left_image": np.zeros((16, 16, 3), np.uint8)}
    assert C.pixel_to_base_point(obs, "robot0_agentview_left", 8, 8) is None


def test_pixel_to_base_point_medians_over_a_patch():
    """One pixel on a silhouette can land on background metres behind; the median is what
    stops a target jumping there."""
    depth = np.full((16, 16), 1.0)
    depth[8, 8] = 9.0                       # a single outlier at the pixel asked for
    obs = {"robot0_agentview_left_depth": depth}
    point = C.pixel_to_base_point(obs, "robot0_agentview_left", 8, 8, patch=2)
    rot, off = C.camera_pose_in_base("robot0_agentview_left")
    assert (rot.T @ (point - off))[2] == pytest.approx(1.0)


def test_cloud_in_base_honours_mask_stride_and_range():
    depth = np.full((32, 32), 1.0)
    depth[:, 16:] = 9.0
    obs = {"robot0_agentview_left_depth": depth}
    near = C.cloud_in_base(obs, "robot0_agentview_left", stride=1, max_range=4.0)
    assert len(near) == 32 * 16                       # the far half is dropped
    assert len(C.cloud_in_base(obs, "robot0_agentview_left", stride=2,
                               max_range=4.0)) == 16 * 8
    mask = np.zeros((32, 32), dtype=bool)
    mask[0:4, 0:4] = True
    assert len(C.cloud_in_base(obs, "robot0_agentview_left", mask=mask, stride=1)) == 16


def _slab(z, n=40, extent=0.5):
    g = np.linspace(-extent, extent, n)
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])


def _grasp(approach, score_pos=(0.0, 0.0, 1.0)):
    mat = np.eye(4)
    z = T.normalize(approach)
    x = T.normalize(np.cross([0.0, 1.0, 0.0], z)) if abs(z[1]) < 0.9 else np.array([1.0, 0, 0])
    mat[:3, 0], mat[:3, 1], mat[:3, 2] = x, np.cross(z, x), z
    mat[:3, 3] = score_pos
    return mat
