"""Compact status and score summaries of Harbor job directories."""
from __future__ import annotations

import html
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import fmean

from .view.scan import read_json, trial_dirs


def _object(value) -> dict:
    return value if isinstance(value, dict) else {}


def _number(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _mean(values) -> float | None:
    numbers = [n for value in values if (n := _number(value)) is not None]
    return fmean(numbers) if numbers else None


def _sum(values) -> float | None:
    numbers = [n for value in values if (n := _number(value)) is not None]
    return math.fsum(numbers) if numbers else None


def _cost(result: dict, trials: list[dict]) -> float | None:
    cost = _number(_object(result.get("stats")).get("cost_usd"))
    if cost is not None:
        return cost
    contexts = []
    for trial in trials:
        if isinstance(trial.get("agent_result"), dict):
            contexts.append(trial["agent_result"])
        else:
            steps = trial.get("step_results")
            if isinstance(steps, list):
                contexts.extend(_object(_object(step).get("agent_result")) for step in steps)
    return _sum(context.get("cost_usd") for context in contexts)


def _trial_scores(path: Path, result: dict) -> dict:
    """Rewards, plus the scalar metrics a task reports beside them.

    reward.json carries the total only; a family that also writes report.json (success
    rates, episode counts, parameter budgets) would otherwise summarise as reward alone."""
    scores = dict(_reward_scores(path, result))
    for key, value in _object(read_json(path / "verifier" / "report.json")).items():
        number = _number(value)
        if number is not None and key not in scores:
            scores[key] = number
    return scores


def _reward_scores(path: Path, result: dict) -> dict:
    rewards = _object(_object(result.get("verifier_result")).get("rewards"))
    if rewards:
        return rewards
    steps = result.get("step_results") or []
    if isinstance(steps, list) and steps:
        final = _object(steps[-1])
        if final.get("step_name") == "develop":
            diagnosis = _object(read_json(path / "steps/develop/verifier/diagnosis.json"))
            if diagnosis.get("phase") != "finished":
                return _object(read_json(path / "steps/evaluate/verifier/reward.json"))
        rewards = _object(_object(final.get("verifier_result")).get("rewards"))
        if rewards:
            return rewards
        name = final.get("step_name")
        if isinstance(name, str) and name not in (".", "..") and "/" not in name:
            path = path / "steps" / name
    elif (path / "steps").is_dir():
        diagnosis = _object(read_json(path / "steps/develop/verifier/diagnosis.json"))
        path = path / "steps" / ("develop" if diagnosis.get("phase") == "finished" else "evaluate")
    return _object(read_json(path / "verifier" / "reward.json"))


def _metric(stats: dict, trials: list[dict], key: str) -> float | None:
    total, count = 0.0, 0
    for evaluation in _object(stats.get("evals")).values():
        histogram = _object(_object(_object(evaluation).get("reward_stats")).get(key))
        for value, names in histogram.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and isinstance(names, list):
                total += number * len(names)
                count += len(names)
    # Harbor's histogram includes only the current result of each retried trial.
    if count:
        return total / count
    return _mean(t.get(key) for t in trials)


def _status(result: dict, trials: list[dict]) -> str:
    stats = _object(result.get("stats"))
    progress = {**_object(result.get("status")), **stats}
    completed = progress.get("n_completed_trials", progress.get("n_trials", 0)) or 0
    errors = progress.get("n_errored_trials", progress.get("n_errors", 0)) or 0
    cancelled = progress.get("n_cancelled_trials", 0) or 0
    total = result.get("n_total_trials") or len(trials)
    if result.get("finished_at"):
        if cancelled:
            return "cancelled"
        if errors or result.get("exception_info"):
            return "failed"
        if total and completed < total and progress:
            return "incomplete"
        if not progress and any(t.get("exception_info") for t in trials):
            return "failed"
        return "completed"
    if progress.get("n_running_trials"):
        return "running"
    if progress.get("n_pending_trials"):
        return "pending"
    if result.get("started_at"):
        return "running"
    if trials:
        if all(t.get("finished_at") or t.get("exception_info") for t in trials):
            if any(t.get("exception_info") for t in trials):
                return "failed"
            return "completed" if len(trials) >= total else "incomplete"
        if any(t.get("started_at") for t in trials):
            return "running"
    return "unknown"


def _identity(config: dict, results: list[dict]) -> tuple[str, str]:
    agents = config.get("agents")
    if isinstance(agents, list) and agents:
        pairs = {
            (a.get("name") or a.get("import_path") or "unknown", a.get("model_name") or "-")
            for a in agents if isinstance(a, dict)
        }
    else:
        pairs = set()
        for result in results:
            agent = _object(_object(result.get("config")).get("agent"))
            info = _object(result.get("agent_info"))
            model_info = _object(info.get("model_info"))
            model = agent.get("model_name") or model_info.get("name") or "-"
            provider = model_info.get("provider")
            if not agent.get("model_name") and provider and model != "-" and "/" not in model:
                model = f"{provider}/{model}"
            name = agent.get("name") or agent.get("import_path") or info.get("name")
            if name:
                pairs.add((name, model))
    if len(pairs) == 1:
        return next(iter(pairs))
    return ("mixed", "mixed") if pairs else ("unknown", "-")


def _metrics(stats: dict, scores: list[dict]) -> dict:
    keys = {key for score in scores for key in score}
    for evaluation in _object(stats.get("evals")).values():
        keys.update(_object(_object(evaluation).get("reward_stats")))
    return {key: _metric(stats, scores, key) for key in sorted(keys)}


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def scan(root: Path) -> list[dict]:
    """Find jobs without descending into trial artifacts or following symlinks."""
    jobs = []
    for directory, children, _ in os.walk(root):
        children[:] = sorted(n for n in children if not n.startswith("."))
        path = Path(directory)
        result = _object(read_json(path / "result.json"))
        config = _object(read_json(path / "config.json"))
        trials = trial_dirs(path)
        if not (trials or "n_total_trials" in result or "job_name" in config
                or (path / "job.log").is_file()):
            continue
        children.clear()
        results = [_object(read_json(t / "result.json")) for t in trials]
        scores = [_trial_scores(t, r) for t, r in zip(trials, results)]
        agent, model = _identity(config, results)
        stats = _object(result.get("stats"))
        metrics = _metrics(stats, scores)
        jobs.append({
            "agent": agent, "model": model,
            "job": path.relative_to(root).as_posix(),
            "status": _status(result, results),
            "cost_usd": _cost(result, results),
            "reward": metrics.get("reward"),
            "success_rate": metrics.get("success_rate"),
            "metrics": metrics,
            "extra": {
                "started_at": result.get("started_at"),
                "finished_at": result.get("finished_at"),
                "n_total_trials": result.get("n_total_trials", len(trials)),
                "progress": {key: value for key, value in stats.items() if key.startswith("n_")},
                "trials": [
                    {"trial": trial.name, "rewards": score}
                    for trial, score in zip(trials, scores)
                ],
            },
        })
    return _json_safe(jobs)


def aggregate(jobs: list[dict]) -> dict:
    return {
        "jobs": len(jobs),
        "cost_usd": _sum(j["cost_usd"] for j in jobs),
        "n_cost": sum(j["cost_usd"] is not None for j in jobs),
        "statuses": dict(sorted(Counter(j["status"] for j in jobs).items())),
        **{key: _mean(j[key] for j in jobs) for key in ("reward", "success_rate")},
        **{f"n_{key}": sum(j[key] is not None for j in jobs) for key in ("reward", "success_rate")},
    }


def _format(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}" if percent else f"{value:.4f}"


def _money(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    for row in [headers, *rows]:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip())


def _stats_row(members: list[dict]) -> list[str]:
    stats = aggregate(members)
    return [
        str(stats["jobs"]),
        ", ".join(f"{status}={count}" for status, count in stats["statuses"].items()),
        _format(stats["reward"]), str(stats["n_reward"]),
        _format(stats["success_rate"], percent=True), str(stats["n_success_rate"]),
        _money(stats["cost_usd"]), str(stats["n_cost"]),
    ]


def tables(jobs: list[dict]) -> list[tuple[str, list[str], list[list[str]]]]:
    groups: dict[str, list[dict]] = {}
    for job in jobs:
        parts = Path(job["job"]).parts
        if len(parts) > 1:
            groups.setdefault(parts[0], []).append(job)
    scopes = [*sorted(groups.items()), ("ALL", jobs)]
    stats_headers = ["Jobs", "Statuses", "Mean reward", "Scored", "Mean success", "Reported",
                     "Total cost (USD)", "Costed"]
    by_pair = []
    for scope, members in scopes:
        pairs: dict[tuple[str, str], list[dict]] = {}
        for job in members:
            pairs.setdefault((job["agent"], job["model"]), []).append(job)
        for (agent, model), runs in sorted(pairs.items()):
            by_pair.append([scope, agent, model, *_stats_row(runs)])
    return [
        ("Group totals", ["Group", *stats_headers],
         [[scope, *_stats_row(members)] for scope, members in scopes]),
        ("By agent and model", ["Group", "Agent", "Model", *stats_headers], by_pair),
        ("Jobs", ["Job", "Agent", "Model", "Status", "Reward", "Success rate", "Cost (USD)"], [
            [j["job"], j["agent"], j["model"], j["status"],
             _format(j["reward"]), _format(j["success_rate"], percent=True), _money(j["cost_usd"])]
            for j in jobs
        ]),
    ]


_NOTES = ("Means include available values only, equally weighted per job. Status is last recorded. "
          "Scored and Reported count jobs contributing reward and success rate, respectively. "
          "Costs sum reported USD values; Costed counts jobs with cost data. "
          "Jobs containing multiple agent/model pairs are labeled mixed.")


def _escape(value: str) -> str:
    return html.escape(value, quote=False).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def markdown(root: Path, sections: list[tuple[str, list[str], list[list[str]]]]) -> str:
    lines = [f"# Job report: {_escape(root.as_posix())}", "", _NOTES, ""]
    for title, headers, rows in sections:
        lines.extend([f"## {title}", ""])
        for row in [headers, ["---"] * len(headers), *rows]:
            lines.append("| " + " | ".join(_escape(value) for value in row) + " |")
        lines.append("")
    return "\n".join(lines)


def render_report(summary_path: Path) -> int:
    """Render solely from a saved summary; no job files are read."""
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: cannot read summary {summary_path}: {exc}") from exc
    if (not isinstance(summary, dict) or summary.get("schema_version") != 1
            or not isinstance(summary.get("runs"), list) or not isinstance(summary.get("root"), str)):
        raise SystemExit("error: expected a version 1 rlebench summary with root and runs")
    jobs = summary["runs"]
    root = Path(summary["root"])
    sections = tables(jobs)
    if not jobs:
        print(f"No jobs found in {root}")
    else:
        for title, headers, rows in sections:
            print(f"\n{title}")
            _table(headers, rows)
        print(f"\n{_NOTES}")
    report = summary_path.with_name("report.md")
    report.write_text(markdown(root, sections))
    print(f"\nSummary: {summary_path}\nReport: {report}")
    return 0


def summarize(root: Path) -> int:
    if root.is_file() and root.suffix == ".json":
        return render_report(root)
    if not root.is_dir():
        raise SystemExit(f"error: not a directory: {root}")
    summary = {"schema_version": 1, "root": root.as_posix(), "runs": scan(root)}
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return render_report(summary_path)
