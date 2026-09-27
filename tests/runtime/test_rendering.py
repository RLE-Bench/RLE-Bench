from types import SimpleNamespace as NS

import numpy as np
import pytest

from harness.backend import Backend, resize


def backend():
    b = Backend.__new__(Backend)
    b.default_resolution, b.sequence, b.video = 128, 3, None
    b.cameras = ("camera",)
    b.raw = {"robot0_joint_pos": np.zeros(7), "object-state": np.ones(3), "robot0_secret": np.ones(2)}
    b.frames = {}
    calls = []
    def render(**kw):
        calls.append(kw)
        rgb = np.tile(np.arange(512, dtype=np.uint16)[:, None, None], (1, 512, 3)).astype(np.uint8)
        z = np.tile(np.linspace(0, 1, 512, dtype=np.float32), (512, 1))
        return (rgb, z) if kw["depth"] else rgb
    model = NS(vis=NS(map=NS(znear=.01, zfar=10)), stat=NS(extent=2),
               camera_name2id=lambda name: 0, cam_fovy=[60])
    b.env = NS(sim=NS(render=render, model=model))
    return b, calls


def test_all_requests_render_once_at_ceiling_and_share_rgb():
    b, calls = backend()
    small = b.observe({"width": 160, "height": 96})
    depth = b.observe({"width": 160, "height": 96, "depth": True})
    big = b.observe({"width": 512, "depth": True})
    assert calls == [dict(camera_name="camera", width=512, height=512, depth=True)]
    np.testing.assert_array_equal(small["obs"]["camera_image"], depth["obs"]["camera_image"])
    assert depth["obs"]["camera_depth"].shape == (96, 160)
    assert depth["obs"]["camera_depth"].dtype == np.float32
    np.testing.assert_array_equal(depth["obs"]["camera_depth"], resize(big["obs"]["camera_depth"],160,96,True))
    assert np.isclose(big["obs"]["camera_depth"][0,0], .02)
    assert np.isclose(big["obs"]["camera_depth"][0,-1], 20, rtol=1e-4)
    k = small["intrinsics"]["camera"]
    assert k[0][0]/k[1][1] == pytest.approx(160/96)
    assert k[0][2] == 79.5 and k[1][2] == 47.5
    assert "object-state" not in small["obs"] and "robot0_secret" not in small["obs"]


def test_proprio_only_and_oversized_requests():
    b, calls = backend()
    assert b.observe({"cameras": []})["obs"].keys() == {"robot0_joint_pos"}
    assert calls == []
    assert b.observe({"width": 2048})["resolution"] == [512,512]


def test_failed_render_is_not_silently_omitted():
    b, _ = backend()
    def fail(**kw):
        raise RuntimeError("EGL failed")
    b.env.sim.render = fail
    with pytest.raises(RuntimeError):
        b.observe({})


@pytest.mark.skipif(not hasattr(Backend, 'render_rgb'), reason='task03 retains observation capture')
def test_video_rgb_matches_agent_rgb_without_poisoning_depth_cache():
    b, calls = backend()
    video_rgb = b.render_rgb('camera')
    assert b.frames == {}
    rgb, depth = b.render('camera')
    np.testing.assert_array_equal(video_rgb, rgb)
    assert depth.shape == (512, 512)
    assert b.render_rgb('camera') is rgb
    assert [c['depth'] for c in calls] == [False, True]
    assert all(c['width'] == c['height'] == 512 for c in calls)


@pytest.mark.skipif(not hasattr(Backend, 'render_rgb'), reason='task03 retains observation capture')
@pytest.mark.parametrize('steps', [0, 1, 2, 5, 6])
def test_evaluation_capture_initial_action_cadence_and_final(tmp_path, steps):
    from rlebench.runtime.media import EvaluationVideo
    from unittest.mock import Mock
    b, calls = backend()
    b.cameras = ('left', 'right', 'wrist')
    b.seed = 0
    b.env.reset = lambda: {}
    b.env.sim.model.vis.global_ = NS(offwidth=512, offheight=512)
    b.env.sim.data = NS(qpos=np.zeros(1))
    b.env.step = lambda action: ({}, 0, False, {})
    b.env.close = lambda: None
    b.after_reset = lambda: None
    b.info = lambda: {}
    b.evidence = lambda: {}
    b.video = EvaluationVideo(tmp_path / 'trial-01.mp4', b.render_rgb, b.cameras)
    b.video.video.close()
    writer = Mock()
    writer.closing.is_set.return_value = False
    b.video.video = writer
    b.reset()
    for _ in range(steps):
        b.act(action=[0])
        b.observe({'cameras': []})
    before = writer.add.call_count
    b.observe({})
    b.observe({})
    assert writer.add.call_count == before
    b.close()
    b.close()
    assert writer.add.call_count == 1 + steps // 2 + steps % 2
    assert all(c['width'] == c['height'] == 512 for c in calls)
    assert writer.add.call_args.args[0].shape == (256, 768, 3)
    assert [c['camera_name'] for c in calls[:3]] == ['left', 'right', 'wrist']


@pytest.mark.skipif(not hasattr(Backend, 'render_rgb'), reason='task03 retains observation capture')
def test_video_capture_failure_is_contained(tmp_path):
    from rlebench.runtime.media import EvaluationVideo
    import json
    b, _ = backend()
    def fail(camera):
        raise RuntimeError('render failed')
    video = EvaluationVideo(tmp_path / 'trial-01.mp4', fail, b.cameras)
    video.capture(b.sequence)
    video.close(b.sequence)
    record = json.loads((tmp_path / 'trial-01.json').read_text())
    assert record['status'] == 'incomplete'
    assert 'render failed' in record['reason']


@pytest.mark.skipif(not hasattr(Backend, 'render_rgb'), reason='task03 keeps its existing defaults')
def test_development_never_creates_recorder(monkeypatch):
    import importlib
    module = importlib.import_module(Backend.__module__)
    monkeypatch.setenv('RLEBENCH_MEDIA', 'true')
    monkeypatch.setattr(module, 'prepare', lambda: None)
    monkeypatch.setattr(Backend, 'create', lambda self: NS())
    def fail(*args):
        raise AssertionError('development created a recorder')
    monkeypatch.setattr(module, 'EvaluationVideo', fail)
    assert Backend(seed=0).video is None
