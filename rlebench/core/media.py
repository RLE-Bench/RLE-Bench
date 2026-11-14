"""Human-facing renders of a scored submission: stills and episode videos.

Evidence for a reader, never input to a score. Every entry point here is
best-effort: a missing encoder, a GL failure or a bad frame is recorded in
the media index and otherwise ignored, so a verifier that renders can never
fail or change a reward because of it. Call sites should still run after
reward.json is on disk, or render from a separate mujoco.Renderer that
never steps physics.

Environment:
  RLEBENCH_MEDIA      "0" disables every writer (default on)
  RLEBENCH_MEDIA_DIR  output directory (default /logs/verifier/media)

Import-time dependencies are stdlib + numpy only. MuJoCo, PIL/matplotlib and
the ffmpeg CLI are looked up when first used.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

DEFAULT_DIR = "/logs/verifier/media"
VIDEO_SIZE = (640, 360)     # (width, height)
VIDEO_FPS = 10
VIDEO_MAX_FRAMES = 3000     # 5 min at VIDEO_FPS; longer episodes are truncated
IMAGE_SIZE = (1280, 960)
CRF = 23


def enabled() -> bool:
    flag = os.environ.get("RLEBENCH_MEDIA", "1").strip().lower()
    return flag not in ("0", "false", "no", "off")


def media_dir(default: str = DEFAULT_DIR) -> Path:
    return Path(os.environ.get("RLEBENCH_MEDIA_DIR", default))


def ffmpeg_exe() -> str | None:
    """The ffmpeg binary: the CLI on PATH, else the one imageio-ffmpeg bundles."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def ffmpeg_available() -> bool:
    return ffmpeg_exe() is not None


def as_rgb8(frame) -> np.ndarray:
    """(H, W, 3|4) of any dtype -> contiguous (H, W, 3) uint8."""
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"expected (H, W, 3|4) image, got {arr.shape}")
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def pad_even(frame: np.ndarray) -> np.ndarray:
    """libx264 needs even dimensions; pad a row or column when a frame is odd."""
    h, w = frame.shape[:2]
    return np.pad(frame, ((0, h % 2), (0, w % 2), (0, 0))) if (h % 2 or w % 2) else frame


def tile(frames, pad_value: int = 0) -> np.ndarray:
    """Concatenate images left to right, padding shorter ones at the bottom."""
    views = [as_rgb8(f) for f in frames if f is not None]
    if not views:
        raise ValueError("no frames to tile")
    height = max(v.shape[0] for v in views)
    padded = [np.pad(v, ((0, height - v.shape[0]), (0, 0), (0, 0)),
                     constant_values=pad_value) for v in views]
    return np.concatenate(padded, axis=1)


def save_png(path, frame) -> Path:
    """Write one RGB frame. Uses PIL, falls back to matplotlib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = as_rgb8(frame)
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except ImportError:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(path, rgb)
    return path


class VideoWriter:
    """Streams rgb24 frames to an mp4 through the ffmpeg CLI.

    `size` may be None: the encoder then opens on the first frame, at that
    frame's size (odd dimensions padded). `add` is cheap to call every tick:
    `every` subsamples, `max_frames` caps the file. Without ffmpeg the writer
    is a no-op and `skipped` says why, so callers need no branch of their own.
    """

    def __init__(self, path, size=None, fps: float = VIDEO_FPS,
                 every: int = 1, max_frames: int = VIDEO_MAX_FRAMES):
        self.path = Path(path)
        self.width = self.height = None
        self.fps = float(fps)
        self.every = max(1, int(every))
        self.max_frames = int(max_frames)
        self.frames = 0
        self._ticks = 0
        self.skipped: str | None = None
        self._proc = None
        self._exe = ffmpeg_exe()
        if self._exe is None:
            self.skipped = "ffmpeg unavailable"
        elif size is not None:
            self._open(int(size[0]), int(size[1]))

    def _open(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            [self._exe, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(CRF),
             str(self.path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)

    @property
    def active(self) -> bool:
        """Frames are still wanted: ffmpeg is present, the cap is not reached,
        and the stream has not failed."""
        return self.skipped is None and self.frames < self.max_frames

    def wants_frame(self) -> bool:
        """True on the ticks a frame is due. Lets callers skip the render itself."""
        due = self.active and self._ticks % self.every == 0
        self._ticks += 1
        return due

    def add(self, frame) -> None:
        """Encode one frame. Callers that use `wants_frame` pass every due frame;
        callers that do not may pass every tick and let `every` subsample."""
        if not self.active:
            return
        rgb = pad_even(as_rgb8(frame))
        if self._proc is None:
            self._open(rgb.shape[1], rgb.shape[0])
        if rgb.shape[:2] != (self.height, self.width):
            raise ValueError(f"frame {rgb.shape[1]}x{rgb.shape[0]} != "
                             f"writer {self.width}x{self.height}")
        self._proc.stdin.write(rgb.tobytes())
        self.frames += 1

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            _, err = proc.communicate(timeout=120)   # closes stdin, drains stderr
            if proc.returncode != 0:
                self.skipped = f"ffmpeg exit {proc.returncode}: {err.decode(errors='replace').strip()}"
        except Exception as exc:  # noqa: BLE001
            proc.kill()
            self.skipped = f"ffmpeg: {exc}"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Recorder:
    """Frames from a simulation loop into one mp4, without the loop having to
    care. `video` is an mp4 path (stream owned here, closed by `close`) or an
    open VideoWriter the caller finishes afterwards, typically one from
    `Media.video`. `every` keeps one frame in N. Any failure -- no ffmpeg, a
    render error, a dead encoder -- marks the stream skipped and ends the
    recording; nothing raises into the loop.
    """

    def __init__(self, video, *, fps: float = VIDEO_FPS, every: int = 1,
                 max_frames: int = VIDEO_MAX_FRAMES):
        self.owns = not isinstance(video, VideoWriter)
        self.stream = (VideoWriter(video, fps=fps, max_frames=max_frames)
                       if self.owns else video)
        self.every = max(1, int(every))
        self._ticks = 0

    @property
    def active(self) -> bool:
        return self.stream.active

    def wants_frame(self) -> bool:
        """True on the ticks a frame is due; call once per loop step."""
        self._ticks += 1
        return self.active and (self._ticks - 1) % self.every == 0

    def add(self, frame) -> None:
        """Encode a frame that is already rendered."""
        try:
            self.stream.add(frame)
        except Exception as exc:  # noqa: BLE001
            self.stream.skipped = f"recorder: {exc}"

    def capture(self, render: Callable[[], np.ndarray]) -> None:
        """One loop step: render and encode only when a frame is due."""
        if self.wants_frame():
            try:
                self.add(render())
            except Exception as exc:  # noqa: BLE001
                self.stream.skipped = f"recorder: {exc}"

    def close(self) -> None:
        if not self.owns:
            return
        self.stream.close()
        if self.stream.frames == 0 and self.stream.path.exists():
            self.stream.path.unlink()


def frame_extent(data, scale: float = 2.2, min_distance: float = 0.8):
    """Camera target and distance from a model's geom positions, so any model
    fills the frame the same way. Returns (lookat, distance)."""
    pos = np.asarray(data.geom_xpos)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    return (lo + hi) / 2.0, max(min_distance, scale * float(np.linalg.norm(hi - lo)))


class MujocoCamera:
    """Offscreen MuJoCo renderer bound to one model, producing rgb8 frames.

    `camera` is a named camera in the model or None for a free camera set by
    `look_at`. Its own Renderer, so it never touches the physics stepping the
    caller does; call `close` when done.
    """

    def __init__(self, model, camera: str | None = None, size=VIDEO_SIZE,
                 headlight: tuple[float, float] | None = None):
        import mujoco
        self._mujoco = mujoco
        width, height = int(size[0]), int(size[1])
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        if headlight is not None:                  # (ambient, diffuse): scenes
            model.vis.headlight.ambient[:] = headlight[0]   # with one light or
            model.vis.headlight.diffuse[:] = headlight[1]   # none need a lift
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = camera
        self.free = mujoco.MjvCamera()
        self.free.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.look_at((0.0, 0.0, 0.5))

    def look_at(self, lookat, distance: float = 3.0, azimuth: float = 135.0,
                elevation: float = -15.0) -> None:
        self.free.lookat[:] = lookat
        self.free.distance = distance
        self.free.azimuth = azimuth
        self.free.elevation = elevation

    def render(self, data) -> np.ndarray:
        cam = self.camera if self.camera is not None else self.free
        self.renderer.update_scene(data, camera=cam)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()


class Media:
    """One verifier's media output: a directory, its files and an index.

    Every method catches its own errors and records them, so a scorer can
    call these unconditionally. `close` writes index.json with the files,
    the skipped reasons and the seconds spent rendering.
    """

    def __init__(self, root=None, on: bool | None = None):
        self.root = Path(root) if root is not None else media_dir()
        self.on = enabled() if on is None else bool(on)
        self.files: list[str] = []
        self.skipped: list[dict] = []
        self.seconds = 0.0

    @classmethod
    def verifier(cls, verifier_dir) -> "Media":
        """The session a verifier entry point opens: <verifier dir>/media."""
        return cls(Path(verifier_dir) / "media")

    def path(self, name: str) -> Path:
        return self.root / name

    def still(self, name: str, model, data, lookat=None, distance: float = 3.0,
              azimuth: float = 135.0, elevation: float = -15.0, *,
              camera: str | None = None, size=IMAGE_SIZE,
              headlight: tuple[float, float] | None = (0.35, 0.5)) -> Path | None:
        """One rendered PNG of `data` from a free camera (or a named one).
        `lookat=None` frames the model's geom extent."""
        if not self.on:
            return None
        t0 = time.monotonic()
        try:
            cam = MujocoCamera(model, camera, size=size, headlight=headlight)
            try:
                if camera is None:
                    if lookat is None:
                        lookat, distance = frame_extent(data)
                    cam.look_at(lookat, distance, azimuth, elevation)
                frame = cam.render(data)
            finally:
                cam.close()
        except Exception as exc:  # noqa: BLE001
            self.skip(name, exc)
            return None
        finally:
            self.seconds += time.monotonic() - t0
        return self.image(name, frame)

    def image(self, name: str, frame) -> Path | None:
        if not self.on:
            return None
        t0 = time.monotonic()
        try:
            path = save_png(self.path(name), frame)
            self.files.append(name)
            return path
        except Exception as exc:  # noqa: BLE001
            self.skip(name, exc)
            return None
        finally:
            self.seconds += time.monotonic() - t0

    def video(self, name: str, **kw) -> VideoWriter | None:
        """Open a writer for `name`; register it on close via `finish`."""
        if not self.on:
            return None
        try:
            return VideoWriter(self.path(name), **kw)
        except Exception as exc:  # noqa: BLE001
            self.skip(name, exc)
            return None

    def finish(self, writer: VideoWriter | None) -> None:
        if writer is None:
            return
        t0 = time.monotonic()
        try:
            writer.close()
        finally:
            self.seconds += time.monotonic() - t0
        name = str(writer.path.relative_to(self.root)) if writer.path.is_relative_to(self.root) \
            else str(writer.path)
        if writer.skipped or writer.frames == 0:
            if writer.path.exists():
                writer.path.unlink()
            self.skip(name, writer.skipped or "no frames")
        else:
            self.files.append(name)

    def run(self, name: str, fn, *args, **kw):
        """Call `fn(self, *args, **kw)` guarded and timed; a raise is recorded
        under `name` and swallowed. For whole render hooks."""
        if not self.on:
            return None
        t0 = time.monotonic()
        try:
            return fn(self, *args, **kw)
        except Exception as exc:  # noqa: BLE001
            self.skip(name, exc)
            return None
        finally:
            self.seconds += time.monotonic() - t0

    def skip(self, name: str, reason) -> None:
        self.skipped.append({"name": name, "reason": str(reason)})

    def report(self) -> dict:
        """Close the session and print its index on one line for the log."""
        index = self.close()
        print(f"media: {json.dumps(index)}", flush=True)
        return index

    @staticmethod
    def export(src, dest) -> list[str]:
        """Copy a finished media session (the files its index names, and the
        index) from a private location next to a reward. Nothing not listed is
        copied, so a recording cut off mid-way leaves no half-written file."""
        src, dest = Path(src), Path(dest)
        index = src / "index.json"
        if not index.is_file():
            return []
        listed = list(json.loads(index.read_text()).get("files", []))
        dest.mkdir(parents=True, exist_ok=True)
        for name in listed:
            shutil.copy2(src / name, dest / name)
        shutil.copy2(index, dest / "index.json")
        return listed

    def close(self) -> dict:
        index = {"enabled": self.on, "files": list(self.files),
                 "skipped": list(self.skipped),
                 "render_seconds": round(self.seconds, 3)}
        if self.on:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                with open(self.root / "index.json", "w") as f:
                    json.dump(index, f, indent=2)
            except Exception as exc:  # noqa: BLE001
                index["index_error"] = str(exc)
        return index

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
