"""Status and metric summaries over complete and partially written job trees."""
import json

import pytest

from rlebench import cli
from rlebench.summarize import aggregate, markdown, scan, tables


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def job(root, name, *, stats=None, finished=True):
    path = root / name
    write(path / "config.json", {"job_name": path.name})
    write(path / "result.json", {
        "n_total_trials": 1, "started_at": "2026-09-09T10:00:00",
        "finished_at": "2026-09-09T11:00:00" if finished else None,
        "stats": stats or {},
    })
    return path


def trial(path, name, reward=None, **result):
    path = path / name
    write(path / "result.json", result)
    (path / "verifier").mkdir()
    if reward is not None:
        write(path / "verifier/reward.json", reward)
    return path


def test_job_histograms_exclude_old_retries_and_weight_trials(tmp_path):
    path = job(tmp_path, "L1/task/codex/model", stats={
        "n_completed_trials": 1, "cost_usd": 12.5,
        "evals": {
            "one": {"reward_stats": {"reward": {"0": ["a"], "1": ["b", "c"]},
                                     "success_rate": {"0.5": ["a", "b"]},
                                     "interaction_steps": {"100": ["a"], "200": ["b"]}}},
            "two": {"reward_stats": {"reward": {"1": ["d"]}}},
        },
    })
    trial(path, "old-retry", {"reward": 0.1, "success_rate": 0.1},
          agent_result={"cost_usd": 100},
          exception_info={"exception_type": "ApiError"},
          agent_info={"name": "codex", "model_info": {"provider": "openai", "name": "test"}})
    rows = scan(tmp_path)
    assert len(rows) == 1
    assert rows[0]["cost_usd"] == 12.5
    assert rows[0]["metrics"]["interaction_steps"] == 150
    assert rows[0]["extra"]["trials"][0]["rewards"]["reward"] == 0.1
    assert rows[0]["reward"] == 0.75
    assert rows[0]["success_rate"] == 0.5
    assert rows[0]["status"] == "completed"
    assert (rows[0]["agent"], rows[0]["model"]) == ("codex", "openai/test")


def test_status_counts_and_missing_metrics(tmp_path):
    variants = [
        ("completed", {"n_completed_trials": 1}, True),
        ("failed", {"n_completed_trials": 1, "n_errored_trials": 1}, True),
        ("cancelled", {"n_completed_trials": 1, "n_cancelled_trials": 1}, True),
        ("running", {"n_running_trials": 1}, False),
        ("pending", {"n_pending_trials": 1}, False),
        ("incomplete", {"n_completed_trials": 0}, True),
    ]
    for name, stats, finished in variants:
        path = job(tmp_path, name, stats=stats, finished=finished)
        if name == "completed":
            trial(path, "trial", {"reward": 0.0}, finished_at="done")
    rows = scan(tmp_path)
    assert {r["job"]: r["status"] for r in rows} == {name: name for name, _, _ in variants}
    stats = aggregate(rows)
    assert stats["jobs"] == 6 and stats["n_reward"] == 1
    assert stats["reward"] == 0 and stats["success_rate"] is None
    assert stats["n_success_rate"] == 0
    assert stats["cost_usd"] is None and stats["n_cost"] == 0


def test_multi_step_uses_final_verifier_and_not_artifacts(tmp_path):
    path = job(tmp_path, "run")
    t = trial(path, "trial", finished_at="done", step_results=[
        {"step_name": "develop", "agent_result": {"cost_usd": 2.5},
         "verifier_result": {"rewards": {"reward": 1}}},
        {"step_name": "evaluate", "agent_result": {"cost_usd": 1.25}},
    ])
    write(t / "steps/develop/verifier/reward.json", {"reward": 1})
    write(t / "steps/evaluate/verifier/reward.json", {"reward": 0.4, "success_rate": 0.2,
                                                     "diagnostics": {"episodes": [1, 2]}})
    write(t / "steps/evaluate/artifacts/fake/result.json", {"n_total_trials": 99})
    write(t / "steps/evaluate/artifacts/fake/verifier/reward.json", {"reward": 1})
    rows = scan(tmp_path)
    assert len(rows) == 1
    assert rows[0]["extra"]["trials"][0]["rewards"]["diagnostics"] == {"episodes": [1, 2]}
    assert rows[0]["cost_usd"] == 3.75
    assert rows[0]["reward"] == 0.4 and rows[0]["success_rate"] == 0.2
    write(t / "result.json", {"verifier_result": {"rewards": {"reward": 0.6}}})
    assert scan(tmp_path)[0]["reward"] == 0.6
    (t / "steps/evaluate/verifier/reward.json").unlink()
    write(t / "result.json", {"step_results": [
        {"step_name": "develop", "verifier_result": {"rewards": {"reward": 1}}},
    ]})
    assert scan(tmp_path)[0]["reward"] is None
    write(t / "result.json", {"agent_result": {"cost_usd": 0.0}, "step_results": [
        {"agent_result": {"cost_usd": 100}},
    ]})
    assert scan(tmp_path)[0]["cost_usd"] == 0.0
    write(t / "result.json", {"agent_result": {"cost_usd": float("nan")}})
    assert scan(tmp_path)[0]["cost_usd"] is None


def test_malformed_live_files_and_empty_directories(tmp_path, capsys):
    path = tmp_path / "run"
    write(path / "config.json", {"job_name": "run"})
    (path / "result.json").write_text('{"stats":')
    t = trial(path, "trial")
    write(t / "result.json", [])
    write(t / "verifier/reward.json", {"reward": float("nan"), "success_rate": "unknown"})
    rows = scan(tmp_path)
    assert len(rows) == 1
    assert {key: rows[0][key] for key in ("job", "agent", "model", "status", "reward", "success_rate", "cost_usd")} == {
        "job": "run", "agent": "unknown", "model": "-", "status": "unknown",
        "reward": None, "success_rate": None, "cost_usd": None,
    }
    assert cli.main(["summarize", str(tmp_path)]) == 0
    raw = (tmp_path / "summary.json").read_text()
    assert "NaN" not in raw
    assert json.loads(raw)["runs"][0]["extra"]["trials"][0]["rewards"]["reward"] is None
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "loop").symlink_to(tmp_path, target_is_directory=True)
    assert cli.main(["summarize", str(empty)]) == 0
    assert "No jobs found" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not a directory"):
        cli.main(["summarize", str(tmp_path / "missing")])


def test_cli_groups_jobs_without_averaging_group_means(tmp_path, capsys, monkeypatch):
    for name, reward, success, agent, model in [
        ("L1/a", 0, 0, "codex", "openai/first"),
        ("L1/b", 1, 1, "codex", "openai/second"),
        ("L2/a", 1, None, "claude-code", "openai/first"),
    ]:
        path = job(tmp_path, name)
        write(path / "config.json", {"job_name": path.name,
                                    "agents": [{"name": agent, "model_name": model}]})
        trial(path, "trial", {"reward": reward, "success_rate": success}, finished_at="done",
              agent_result={"cost_usd": {"L1/a": 2.5, "L1/b": 3.0, "L2/a": None}[name]})
    assert cli.main(["summarize", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "L1/a" in output and "L1/b" in output and "L2/a" in output
    overall = next(line for line in output.splitlines() if line.startswith("ALL"))
    assert "completed=3" in overall and "0.6667" in overall and "50.0%" in overall
    assert "Total cost (USD)" in output and "5.50" in overall
    assert aggregate(scan(tmp_path))["n_cost"] == 2
    assert "equally weighted per job" in output
    report = (tmp_path / "report.md").read_text()
    summary_path = tmp_path / "summary.json"
    saved = json.loads(summary_path.read_text())
    assert saved["schema_version"] == 1 and saved["root"] == str(tmp_path)
    assert saved["runs"] == scan(tmp_path)
    assert report == markdown(tmp_path, tables(saved["runs"]))
    assert saved["runs"][0]["extra"]["started_at"] == "2026-09-09T10:00:00"
    assert "## By agent and model" in report and "## Jobs" in report
    assert "Cost (USD)" in report and "| 5.50 | 2 |" in report
    assert "| ALL | codex | openai/first | 1 | completed=1 | 0.0000 |" in report
    assert "| L1 | codex | openai/second | 1 | completed=1 | 1.0000 |" in report
    assert "| ALL | claude-code | openai/first | 1 | completed=1 | 1.0000 |" in report
    assert len(scan(tmp_path)) == 3  # report.md is not a job
    rows = scan(tmp_path)
    rows.append({**rows[0], "job": "L2/repeat", "reward": 1.0})
    pair_rows = tables(rows)[1][2]
    assert [row for row in pair_rows if row[:3] == ["ALL", "codex", "openai/first"]][0][-2:] == ["5.00", "2"]
    assert [row for row in pair_rows if row[:3] == ["ALL", "codex", "openai/first"]][0][3:6] == [
        "2", "completed=2", "0.5000"]
    escaped = markdown(tmp_path, [("Jobs", ["Model"], [["vendor/m|<tag>\nvalue"]])])
    assert "vendor/m\\|&lt;tag&gt; value" in escaped
    assert cli.main(["summarize", str(tmp_path / "L1/a")]) == 0
    assert "completed=1" in capsys.readouterr().out
    assert (tmp_path / "L1/a/report.md").is_file()
    assert cli.parse(["summarize"]).folder == "jobs"

    # A saved snapshot can render independently, even after its source runs change.
    from rlebench import summarize as summarizer
    def no_scan(_):
        pytest.fail("rendering a saved summary must not rescan jobs")
    monkeypatch.setattr(summarizer, "scan", no_scan)
    saved["runs"][0]["reward"] = 0.123456789
    saved["runs"][0]["cost_usd"] = 7.25
    snapshot = tmp_path / "snapshot" / "summary.json"
    write(snapshot, saved)
    assert cli.main(["summarize", str(snapshot)]) == 0
    regenerated = snapshot.with_name("report.md").read_text()
    assert regenerated == markdown(tmp_path, tables(saved["runs"]))
    assert "0.1235" in regenerated and "7.25" in regenerated
    assert json.loads(snapshot.read_text())["runs"][0]["reward"] == 0.123456789
    bad = tmp_path / "invalid.json"
    write(bad, {"schema_version": 2, "runs": []})
    with pytest.raises(SystemExit, match="version 1"):
        cli.main(["summarize", str(bad)])


def test_completed_single_step_development_reward(tmp_path):
    path = job(tmp_path, 'single-step')
    t = trial(path, 'trial', finished_at='done', step_results=[{'step_name': 'develop'}])
    write(t / 'steps/develop/verifier/diagnosis.json', {'phase': 'finished'})
    write(t / 'steps/develop/verifier/reward.json', {'reward': .8})
    assert scan(tmp_path)[0]['reward'] == .8
    write(t / 'result.json', {})
    assert scan(tmp_path)[0]['reward'] == .8
