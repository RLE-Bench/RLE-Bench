"""`rlebench view`: the jobs browser's scanner, API and CLI, against a fake tree."""
from __future__ import annotations

import json

import numpy as np
import pytest

from rlebench import cli
from rlebench.view import scan, server

PNG = None


def _png() -> bytes:
    global PNG
    if PNG is None:
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.full((12, 16, 3), 200, dtype=np.uint8)).save(buf, format="PNG")
        PNG = buf.getvalue()
    return PNG


def _trial(job_dir, name, *, reward=0.5, media=True, trajectory=False, finished="2026-09-05T10:00:00Z"):
    t = job_dir / name
    (t / "verifier").mkdir(parents=True)
    (t / "config.json").write_text("{}")
    (t / "result.json").write_text(json.dumps({
        "task_name": f"rlebench/{job_dir.name}", "trial_name": name,
        "agent_info": {"name": "oracle", "version": "1.0.0", "model_info": None},
        "started_at": "2026-09-05T09:59:00Z", "finished_at": finished}))
    (t / "verifier" / "reward.json").write_text(json.dumps({"reward": reward, "gated": 0, "cp_A": reward}))
    (t / "verifier" / "report.json").write_text(json.dumps({"stages": {"a": reward}, "checkpoints": {"A.x": reward}}))
    (t / "verifier" / "test-stdout.txt").write_text("line 1\nline 2\n")
    if media:
        m = t / "verifier" / "media"
        m.mkdir()
        (m / "view.png").write_bytes(_png())
        (m / "ep0.mp4").write_bytes(bytes(range(256)) * 4)          # 1024 bytes stand in for a clip
        (m / "index.json").write_text(json.dumps({"enabled": True, "files": ["view.png", "ep0.mp4"],
                                                  "skipped": [{"name": "ep1.mp4", "reason": "no GL"}],
                                                  "render_seconds": 0.5}))
    if trajectory:
        (t / "agent").mkdir()
        (t / "agent" / "trajectory.json").write_text(json.dumps({
            "schema_version": "ATIF-v1", "agent": {"name": "codex", "model_name": "m"},
            "steps": [{"step_id": 1, "timestamp": "2026-09-05T09:59:01Z", "source": "user", "message": "do it"},
                      {"step_id": 2, "timestamp": "2026-09-05T09:59:02Z", "source": "agent", "message": "ok",
                       "tool_calls": [{"function_name": "exec", "arguments": {"cmd": "ls"}}], "observation": "files"}],
            "final_metrics": {"total_steps": 2, "total_cost_usd": 0.01}}))
    return t


@pytest.fixture
def jobs(tmp_path):
    root = tmp_path / "jobs"
    flat = root / "demo"
    flat.mkdir(parents=True)
    (flat / "result.json").write_text(json.dumps({"finished_at": "2026-09-05T10:00:00Z"}))
    _trial(flat, "demo__aaa", reward=0.7)
    nested = root / "task08" / "codex" / "gpt"                      # jobs/<family>/<agent>/<model>
    nested.mkdir(parents=True)
    _trial(nested, "task08__bbb", reward=0.0, media=False, trajectory=True, finished="2026-09-05T12:00:00Z")
    multi = root / "multi"
    multi.mkdir()
    t = multi / "multi__ccc"
    (t / "steps" / "develop" / "verifier").mkdir(parents=True)
    (t / "steps" / "evaluate" / "verifier" / "media").mkdir(parents=True)
    (t / "steps" / "evaluate" / "agent").mkdir()
    (t / "debug" / "episodes").mkdir(parents=True)
    (t / "config.json").write_text("{}")
    (t / "result.json").write_text(json.dumps({
        "task_name": "rlebench/multi", "agent_info": {"name": "codex", "version": "1", "model_info": {"name": "m"}},
        "started_at": "2026-09-05T09:00:00Z", "finished_at": "2026-09-05T11:00:00Z",
        "step_results": [{"step_name": "develop"}, {"step_name": "evaluate"}]}))
    (t / "steps" / "develop" / "verifier" / "reward.json").write_text(json.dumps({"reward": 0.0}))
    (t / "steps" / "develop" / "verifier" / "test-stdout.txt").write_text("develop out\n")
    (t / "steps" / "evaluate" / "verifier" / "reward.json").write_text(json.dumps({"reward": 0.4}))
    (t / "steps" / "evaluate" / "verifier" / "test-stdout.txt").write_text("evaluate out\n")
    (t / "steps" / "evaluate" / "verifier" / "media" / "trial000.mp4").write_bytes(b"clip")
    (t / "steps" / "evaluate" / "verifier" / "media" / "index.json").write_text(json.dumps({"files": ["trial000.mp4"], "skipped": []}))
    (t / "steps" / "evaluate" / "agent" / "trajectory.json").write_text(json.dumps({"steps": [{"step_id": 1, "source": "agent", "message": "hi"}]}))
    (t / "debug" / "episodes" / "evaluation_ep0000.mp4").write_bytes(b"dbg")
    (root / "notes").mkdir()                                          # a stray dir is not a job
    return root


def test_scan_finds_flat_and_nested_jobs_newest_first(jobs):
    found = scan.scan_jobs(jobs)
    assert [j["job"] for j in found] == ["task08/codex/gpt", "multi", "demo"]
    demo = found[2]["trials"][0]
    assert (demo["reward"], demo["media"], demo["media_skipped"], demo["trajectory"]) == (0.7, 2, 1, False)
    assert found[0]["trials"][0]["trajectory"] is True and found[0]["trials"][0]["media"] == 0


def test_paths_cannot_leave_the_tree(jobs):
    assert scan.resolve_under(jobs, "demo", "demo__aaa") == (jobs / "demo" / "demo__aaa").resolve()
    for bad in ("../..", "demo/../../etc", "/etc", "a//b"):
        assert scan.resolve_under(jobs, bad, "x") is None


@pytest.fixture
def client(jobs):
    from fastapi.testclient import TestClient
    return TestClient(server.create_app(jobs))


def test_multi_step_trials_default_to_the_scored_step(client):
    t = client.get("/api/jobs/multi/trials/multi__ccc").json()
    assert (t["steps"], t["step"], t["reward"]) == (["develop", "evaluate"], "evaluate", 0.4)
    assert t["step_rewards"] == {"develop": 0.0, "evaluate": 0.4}
    assert t["media_index"]["files"] == ["trial000.mp4"] and t["trajectory"] is True
    assert t["debug_episode_files"] == ["evaluation_ep0000.mp4"]
    dev = client.get("/api/jobs/multi/trials/multi__ccc?step=develop").json()
    assert (dev["step"], dev["reward"], dev["media_index"], dev["trajectory"]) == ("develop", 0.0, None, False)
    assert client.get("/api/jobs/multi/trials/multi__ccc/verifier-output?step=develop").json()["stdout"] == "develop out\n"
    assert client.get("/api/jobs/multi/trials/multi__ccc/verifier-output").json()["stdout"] == "evaluate out\n"
    assert client.get("/api/jobs/multi/trials/multi__ccc/media/trial000.mp4").content == b"clip"
    assert client.get("/api/jobs/multi/trials/multi__ccc/debug/evaluation_ep0000.mp4").content == b"dbg"
    for bad in ("?step=nope", "/media/trial000.mp4?step=develop", "/debug/..%2F..%2Fresult.json"):
        assert client.get("/api/jobs/multi/trials/multi__ccc" + bad).status_code == 404, bad


def test_api_lists_jobs_and_details_a_trial(client):
    listed = client.get("/api/jobs").json()
    assert [j["job"] for j in listed] == ["task08/codex/gpt", "multi", "demo"]
    d = client.get("/api/jobs/demo/trials/demo__aaa").json()
    assert d["reward"] == 0.7 and d["media_index"]["files"] == ["view.png", "ep0.mp4"]
    assert d["report"]["checkpoints"] == {"A.x": 0.7} and d["stdout_lines"] == 2
    assert {f["path"] for f in d["files"]} >= {"verifier/reward.json", "verifier/media/view.png"}
    nested = client.get("/api/jobs/task08/codex/gpt/trials/task08__bbb").json()
    assert nested["trajectory"] is True and nested["media_index"] is None


def test_api_serves_media_with_ranges_and_trajectory(client):
    png = client.get("/api/jobs/demo/trials/demo__aaa/media/view.png")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png" and png.content == _png()
    part = client.get("/api/jobs/demo/trials/demo__aaa/media/ep0.mp4", headers={"Range": "bytes=256-511"})
    assert part.status_code == 206 and part.content == bytes(range(256))
    assert client.get("/api/jobs/demo/trials/demo__aaa/media").json()["skipped"][0]["name"] == "ep1.mp4"
    assert client.get("/api/jobs/task08/codex/gpt/trials/task08__bbb/media").json()["files"] == []
    traj = client.get("/api/jobs/task08/codex/gpt/trials/task08__bbb/trajectory").json()
    assert [s["source"] for s in traj["steps"]] == ["user", "agent"]
    assert client.get("/api/jobs/demo/trials/demo__aaa/trajectory").status_code == 404
    out = client.get("/api/jobs/demo/trials/demo__aaa/verifier-output").json()
    assert out["stdout"].startswith("line 1") and out["stderr"] is None


def test_api_refuses_paths_outside_the_tree(client, jobs):
    (jobs / "secret.txt").write_text("no")
    for url in ("/api/jobs/demo/trials/demo__aaa/media/..%2F..%2F..%2Fsecret.txt",
                "/api/jobs/..%2F/trials/demo__aaa", "/api/jobs/demo/trials/nope",
                "/api/jobs/notes/trials/x"):
        assert client.get(url).status_code == 404, url
    page = client.get("/")
    assert page.status_code == 200 and "rlebench view" in page.text
    assert "[hidden] { display: none !important; }" in page.text   # the lightbox starts closed
    assert 'id="q"' in page.text and 'id="media-only"' in page.text   # the rail filter


def test_cli_view_defaults_follow_harbor_view(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_view", lambda args: seen.append(args) or 0)
    assert cli.main(["view"]) == 0
    assert (seen[-1].folder, seen[-1].port, seen[-1].host) == ("jobs", "8080-8089", "127.0.0.1")
    assert cli.main(["view", "runs", "-p", "9000"]) == 0
    assert (seen[-1].folder, server.parse_ports(seen[-1].port)) == ("runs", (9000, 9000))
    assert server.parse_ports("8080-8089") == (8080, 8089)


def test_single_step_defaults_to_develop(client, jobs):
    t = jobs / 'single' / 'single__aaa'
    v = t / 'steps/develop/verifier'
    v.mkdir(parents=True)
    (t / 'config.json').write_text('{}')
    (t / 'result.json').write_text(json.dumps({'step_results': [{'step_name': 'develop'}]}))
    (v / 'reward.json').write_text(json.dumps({'reward': .8}))
    result = client.get('/api/jobs/single/trials/single__aaa').json()
    assert (result['steps'], result['step'], result['reward']) == (['develop'], 'develop', .8)
