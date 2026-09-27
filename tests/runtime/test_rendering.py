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
        return rgb, z
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
