"""Best-effort video encoding; a bounded queue keeps it off the control path."""
import atexit
from pathlib import Path
import queue
import subprocess
import threading


class Video:
    def __init__(self, path):
        self.path = Path(path)
        self.queue = queue.Queue(maxsize=2)
        self.thread = threading.Thread(target=self._encode, daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def add(self, frame):
        if not self.thread.is_alive():
            return
        try:
            self.queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def close(self):
        try:
            self.queue.put(None, timeout=.1)
        except queue.Full:
            pass
        self.thread.join(timeout=.5)

    def _encode(self):
        proc = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            while True:
                frame = self.queue.get()
                if frame is None:
                    break
                if proc is None:
                    h, w = frame.shape[:2]
                    proc = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "5",
                        "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "ultrafast",
                        "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov", str(self.path)],
                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                proc.stdin.write(frame.tobytes())
        except Exception:
            pass
        finally:
            if proc:
                try:
                    proc.stdin.close()
                    proc.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    proc.kill()
                    proc.wait()
