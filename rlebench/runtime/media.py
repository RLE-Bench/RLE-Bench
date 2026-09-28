"""Private, best-effort evaluation recordings with bounded encoding and shutdown."""
import atexit
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

from rlebench.core.media import ffmpeg_exe


def evaluation_enabled():
    return os.environ.get("RLEBENCH_MEDIA", "false").strip().lower() in ("true", "1")


class Video:
    def __init__(self, path, fps=5):
        self.path = Path(path)
        self.fps = fps
        self.queue = queue.Queue(maxsize=32)
        self.lock = threading.RLock()
        self.closing = threading.Event()
        self.proc = None
        self.reason = None
        self.closed = False
        self.frames = 0
        self.thread = threading.Thread(target=self._encode, daemon=True)
        self._metadata("recording")
        try:
            self.thread.start()
        except RuntimeError as exc:
            self.fail(f"encoder thread unavailable: {exc}")
        atexit.register(self.close)

    def _metadata(self, status):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            path = self.path.with_suffix(".json")
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(dict(status=status, reason=self.reason, frames=self.frames)))
            temporary.replace(path)
        except OSError:
            self.reason = self.reason or "metadata write failed"

    def fail(self, reason):
        with self.lock:
            self.reason = self.reason or reason
            self.closing.set()
            self._metadata("incomplete")

    def add(self, frame):
        if self.closing.is_set():
            return
        try:
            self.queue.put_nowait(frame.copy())
        except queue.Full:
            self.fail("encoder queue full")
        except Exception as exc:
            self.fail(f"frame copy failed: {type(exc).__name__}")

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.closing.set()
        deadline = time.monotonic() + 2
        if self.thread.ident is not None:
            # Reserve time to reap a stuck encoder and finish thread cleanup.
            self.thread.join(timeout=max(0, deadline - time.monotonic() - 1))
        if self.thread.is_alive():
            self.fail("encoder shutdown timed out")
            with self.lock:
                if self.proc is not None and self.proc.poll() is None:
                    self.proc.kill()
            self.thread.join(timeout=max(0, deadline - time.monotonic()))
        atexit.unregister(self.close)

    def _encode(self):
        try:
            while not self.reason:
                try:
                    frame = self.queue.get(timeout=.05)
                except queue.Empty:
                    if self.closing.is_set():
                        break
                    continue
                with self.lock:
                    if self.reason:
                        break
                    if self.proc is None:
                        exe = ffmpeg_exe()
                        if not exe:
                            raise RuntimeError("ffmpeg unavailable")
                        h, w = frame.shape[:2]
                        self.proc = subprocess.Popen([exe, "-nostdin", "-loglevel", "error", "-y",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(self.fps),
                            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "ultrafast",
                            "-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)],
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.proc.stdin.write(frame.tobytes())
                self.frames += 1
            if self.proc is not None:
                self.proc.stdin.close()
                if self.proc.wait(timeout=2):
                    self.fail("encoder exited unsuccessfully")
            elif not self.reason:
                self.fail("no frames captured")
        except Exception as exc:
            self.fail(f"encoding failed: {type(exc).__name__}: {exc}")
        finally:
            with self.lock:
                if self.proc is not None and self.proc.poll() is None:
                    self.proc.kill()
                self._metadata("incomplete" if self.reason else "complete")
            if self.proc is not None:
                self.proc.wait()


class EvaluationVideo:
    """Capture on the simulator thread; encode CPU-resized RGB on a worker thread."""
    def __init__(self, path, render, cameras):
        self.video = Video(path, fps=10)
        self.render = render
        self.cameras = cameras
        self.first = None
        self.last = None

    def capture(self, sequence, final=False):
        if self.video.closing.is_set() or sequence == self.last:
            return
        if self.first is None:
            self.first = sequence
        if not final and (sequence - self.first) % 2:
            return
        try:
            import numpy as np
            from PIL import Image
            frames = [np.asarray(Image.fromarray(self.render(camera)).resize(
                (256, 256), Image.Resampling.BOX)) for camera in self.cameras]
            self.video.add(np.concatenate(frames, axis=1))
            self.last = sequence
        except Exception as exc:
            self.video.fail(f"capture failed: {type(exc).__name__}: {exc}")

    def close(self, sequence):
        self.capture(sequence, final=True)
        self.video.close()
