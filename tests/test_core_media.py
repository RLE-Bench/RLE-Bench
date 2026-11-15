"""rlebench.core.media: renders are evidence, never a failure mode."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from rlebench.core import media


def _frame(h=36, w=64, v=0.5):
    return np.full((h, w, 3), v, dtype=np.float32)


def test_as_rgb8_and_tile_normalise_shapes():
    rgb = media.as_rgb8(_frame(v=0.5))
    assert rgb.dtype == np.uint8 and rgb.shape == (36, 64, 3) and rgb[0, 0, 0] == 127
    rgba = np.zeros((36, 64, 4), dtype=np.uint8)
    tiled = media.tile([rgb, rgba, np.zeros((20, 10, 3), dtype=np.uint8)])
    assert tiled.shape == (36, 64 + 64 + 10, 3)
    with pytest.raises(ValueError):
        media.as_rgb8(np.zeros((36, 64)))


def test_video_writer_without_ffmpeg_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "ffmpeg_exe", lambda: None)
    w = media.VideoWriter(tmp_path / "ep.mp4")
    assert not w.active and w.skipped == "ffmpeg unavailable"
    w.add(_frame(360, 640))
    w.close()
    assert w.frames == 0 and not (tmp_path / "ep.mp4").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_video_writer_sizes_itself_from_the_first_frame(tmp_path):
    with media.VideoWriter(tmp_path / "ep.mp4", fps=10) as w:     # no size given
        assert w.active and w.width is None
        w.add(_frame(35, 63))                                     # odd: padded to 36x64
        w.add(_frame(35, 63))
    assert (w.width, w.height, w.frames, w.skipped) == (64, 36, 2, None)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_recorder_keeps_cadence_owns_or_borrows_and_never_raises(tmp_path):
    owned = media.Recorder(tmp_path / "own.mp4", fps=10, every=3)
    for i in range(9):
        owned.capture(lambda: _frame())
    owned.close()
    assert owned.stream.frames == 3 and (tmp_path / "own.mp4").stat().st_size > 0

    m = media.Media(tmp_path / "media")
    borrowed = media.Recorder(m.video("ep.mp4"), every=2)
    borrowed.capture(lambda: _frame())
    borrowed.capture(lambda: _frame())
    borrowed.capture(lambda: 1 / 0)          # a render that dies ends the recording
    borrowed.capture(lambda: _frame())
    borrowed.close()                          # borrowed: left to the session
    assert not borrowed.active and "recorder:" in borrowed.stream.skipped
    m.finish(borrowed.stream)
    assert [s["name"] for s in m.close()["skipped"]] == ["ep.mp4"]


def test_export_copies_only_indexed_files(tmp_path):
    src, dest = tmp_path / "private", tmp_path / "public"
    src.mkdir()
    (src / "a.mp4").write_bytes(b"a")
    (src / "half.mp4").write_bytes(b"cut off")
    (src / "index.json").write_text(json.dumps({"files": ["a.mp4"], "skipped": []}))
    assert media.Media.export(src, dest) == ["a.mp4"]
    assert sorted(p.name for p in dest.iterdir()) == ["a.mp4", "index.json"]
    assert media.Media.export(tmp_path / "absent", dest) == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_video_writer_subsamples_and_caps(tmp_path):
    with media.VideoWriter(tmp_path / "ep.mp4", size=(64, 36), fps=10,
                           every=2, max_frames=3) as w:
        for _ in range(20):
            if w.wants_frame():
                w.add(_frame())
    assert w.frames == 3 and w.skipped is None
    assert (tmp_path / "ep.mp4").stat().st_size > 0


def test_media_records_files_skips_and_index(tmp_path, monkeypatch):
    monkeypatch.setenv("RLEBENCH_MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setattr(media, "ffmpeg_exe", lambda: None)
    with media.Media() as m:
        assert m.image("still.png", _frame()) is not None
        assert m.image("bad.png", np.zeros(3)) is None
        m.finish(m.video("ep.mp4"))

        def boom(media_):
            raise RuntimeError("no GL")
        m.run("hook", boom)
    index = json.loads((tmp_path / "media" / "index.json").read_text())
    assert index["files"] == ["still.png"]
    assert [s["name"] for s in index["skipped"]] == ["bad.png", "ep.mp4", "hook"]
    assert "no GL" in index["skipped"][2]["reason"]


def test_media_off_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("RLEBENCH_MEDIA", "0")
    m = media.Media(tmp_path / "media")
    assert m.image("still.png", _frame()) is None and m.video("ep.mp4") is None
    assert m.run("hook", lambda media_: 1) is None
    assert m.close()["enabled"] is False
    assert not (tmp_path / "media").exists()


def test_still_frames_a_model_and_records_it(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    m = media.Media(tmp_path)
    if m.still("box.png", model, data, size=(64, 36)) is None:      # lookat=None: auto-framed
        pytest.skip(f"no offscreen GL: {m.skipped}")
    lookat, distance = media.frame_extent(data)
    assert np.allclose(lookat, 0) and distance == 0.8                 # a point extent -> the floor
    assert m.close()["files"] == ["box.png"] and (tmp_path / "box.png").stat().st_size > 0


def test_mujoco_camera_renders_a_frame():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 3"/><geom type="box" size=".1 .1 .1"/>'
        '<camera name="cam" pos="1 1 1" xyaxes="-1 1 0 -1 -1 2"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        cam = media.MujocoCamera(model, "cam", size=(64, 36))
    except Exception as exc:  # no GL context on this host
        pytest.skip(f"no offscreen GL: {exc}")
    try:
        frame = cam.render(data)
        free = media.MujocoCamera(model, None, size=(64, 36))
        free.look_at((0, 0, 0), distance=1.0)
        frame2 = free.render(data)
        free.close()
    finally:
        cam.close()
    assert frame.shape == (36, 64, 3) and frame.dtype == np.uint8 and frame.max() > 0
    assert frame2.shape == (36, 64, 3)
