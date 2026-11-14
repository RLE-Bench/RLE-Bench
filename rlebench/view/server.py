"""`rlebench view`: browse a jobs tree in the browser.

The same hierarchy and tabs as `harbor view` -- jobs, trials, trajectory,
verifier output, reward, files -- plus a Media tab for the stills and videos
the verifiers write under verifier/media/ (see rlebench.core.media). One
FastAPI app serving JSON under /api and a single-file frontend at /.
"""
from __future__ import annotations

import mimetypes
import socket
from pathlib import Path

from . import scan

STATIC = Path(__file__).parent / "static"


def parse_ports(spec: str) -> tuple[int, int]:
    """'8080' or '8080-8089' -> (first, last), harbor view's convention."""
    lo, _, hi = spec.partition("-")
    return int(lo), int(hi or lo)


def free_port(host: str, ports: tuple[int, int]) -> int:
    for port in range(ports[0], ports[1] + 1):
        with socket.socket() as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"error: no free port in {ports[0]}-{ports[1]} on {host}")


def create_app(root: Path):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse

    root = root.resolve()
    app = FastAPI(title="rlebench view", docs_url=None, redoc_url=None)

    def trial_dir(job: str, trial: str) -> Path:
        if scan.safe_segments(trial) is None or "/" in trial:
            raise HTTPException(404, "no such trial")
        path = scan.resolve_under(root, job, trial)
        if path is None or not scan.is_trial(path):
            raise HTTPException(404, "no such trial")
        return path

    def step_dir(job: str, trial: str, step: str | None) -> Path:
        """The directory a step's agent/ and verifier/ live in: the trial itself
        for single-step tasks, steps/<step> otherwise (the scored step when unnamed)."""
        t = trial_dir(job, trial)
        path = scan.step_root(t, step or scan.default_step(t))
        if path is None:
            raise HTTPException(404, "no such step")
        return path

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        return scan.scan_jobs(root)

    def send(path: Path, name: str):
        if not path.is_file():
            raise HTTPException(404, "no such file")
        media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type)   # honours Range for video

    def one_segment(name: str) -> str:
        if scan.safe_segments(name) is None or "/" in name:
            raise HTTPException(404, "no such file")
        return name

    @app.get("/api/jobs/{job:path}/trials/{trial}/trajectory")
    def trajectory(job: str, trial: str, step: str | None = None) -> dict:
        data = scan.read_json(step_dir(job, trial, step) / "agent" / "trajectory.json")
        if data is None:
            raise HTTPException(404, "no agent/trajectory.json")
        return data

    @app.get("/api/jobs/{job:path}/trials/{trial}/verifier-output")
    def verifier_output(job: str, trial: str, step: str | None = None) -> dict:
        d = step_dir(job, trial, step) / "verifier"
        return {"stdout": scan.read_text(d / "test-stdout.txt"),
                "stderr": scan.read_text(d / "test-stderr.txt")}

    @app.get("/api/jobs/{job:path}/trials/{trial}/media")
    def media(job: str, trial: str, step: str | None = None) -> dict:
        index = scan.read_json(step_dir(job, trial, step) / "verifier" / "media" / "index.json")
        return index or {"enabled": False, "files": [], "skipped": []}

    @app.get("/api/jobs/{job:path}/trials/{trial}/media/{name}")
    def media_file(job: str, trial: str, name: str, step: str | None = None):
        return send(step_dir(job, trial, step) / "verifier" / "media" / one_segment(name), name)

    @app.get("/api/jobs/{job:path}/trials/{trial}")
    def trial(job: str, trial: str, step: str | None = None) -> dict:
        t = trial_dir(job, trial)
        if step is not None and scan.step_root(t, step) is None:
            raise HTTPException(404, "no such step")
        detail = scan.trial_detail(t, step)
        detail["job"] = job
        return detail

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", media_type="text/html")

    return app


def serve(root: Path, host: str, ports: tuple[int, int]) -> int:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"error: {exc}; `make install` provides the viewer's dependencies")
    if not root.is_dir():
        raise SystemExit(f"error: {root} is not a directory")
    port = free_port(host, ports)
    print(f"rlebench view: {root.resolve()} at http://{host}:{port}  (Ctrl-C stops)", flush=True)
    uvicorn.run(create_app(root), host=host, port=port, log_level="warning")
    return 0
