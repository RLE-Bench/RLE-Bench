"""Read a Harbor jobs tree the way `harbor view` does, plus what RLE-Bench's
verifiers add next to reward.json: the media index and the files it names.

Pure functions over the filesystem; the server is a thin layer on top. Every
reader tolerates a missing or malformed file, since a tree under inspection is
often a run that went wrong.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

TRIAL_MARKERS = ("result.json", "config.json")
TRIAL_CONTENT = ("verifier", "agent", "trial.log", "steps")
MAX_DEPTH = 5


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def is_trial(d: Path) -> bool:
    return (d.is_dir() and any((d / m).is_file() for m in TRIAL_MARKERS)
            and any((d / c).exists() for c in TRIAL_CONTENT))


def trial_dirs(job_dir: Path) -> list[Path]:
    try:
        return sorted(d for d in job_dir.iterdir() if is_trial(d))
    except OSError:
        return []


def safe_segments(rel: str) -> list[str] | None:
    """`a/b/c` -> ['a', 'b', 'c'], or None when any part could leave the tree."""
    parts = rel.split("/") if rel else []
    for p in parts:
        if p in ("", ".", "..") or "\\" in p or os.sep in p:
            return None
    return parts


def resolve_under(root: Path, *rel: str) -> Path | None:
    """root/rel..., or None when the result escapes root."""
    parts: list[str] = []
    for r in rel:
        segs = safe_segments(r)
        if segs is None:
            return None
        parts.extend(segs)
    target = (root / Path(*parts)).resolve() if parts else root.resolve()
    return target if target == root.resolve() or target.is_relative_to(root.resolve()) else None


# ---------------------------------------------------------------------------
# steps: a multi-step task keeps each step's agent/ and verifier/ under
# steps/<name>/; the scored one is the last step that wrote a reward
# ---------------------------------------------------------------------------
def steps(trial_dir: Path) -> list[str]:
    root = trial_dir / "steps"
    if not root.is_dir():
        return []
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    result = read_json(trial_dir / "result.json") or {}
    order = [s.get("step_name") for s in result.get("step_results") or [] if isinstance(s, dict)]
    return [n for n in order if n in names] + [n for n in names if n not in order]


def default_step(trial_dir: Path) -> str | None:
    names = steps(trial_dir)
    scored = [n for n in names if (trial_dir / "steps" / n / "verifier" / "reward.json").is_file()]
    return (scored or names or [None])[-1]


def step_root(trial_dir: Path, step: str | None) -> Path | None:
    """trial_dir, or trial_dir/steps/<step>; None when the step does not exist."""
    if step is None:
        return trial_dir
    if safe_segments(step) is None or "/" in step:
        return None
    root = trial_dir / "steps" / step
    return root if root.is_dir() else None


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def trial_summary(trial_dir: Path, step: str | None = None) -> dict:
    result = read_json(trial_dir / "result.json") or {}
    names = steps(trial_dir)
    step = step or default_step(trial_dir)
    root = step_root(trial_dir, step) or trial_dir
    reward = read_json(root / "verifier" / "reward.json") or {}
    index = read_json(root / "verifier" / "media" / "index.json") or {}
    agent = result.get("agent_info") or {}
    model = (agent.get("model_info") or {}).get("name")
    exc = result.get("exception_info")
    return {
        "trial": trial_dir.name,
        "task": result.get("task_name"),
        "agent": agent.get("name"),
        "agent_version": agent.get("version"),
        "model": model,
        "started": result.get("started_at"),
        "finished": result.get("finished_at"),
        "reward": reward.get("reward"),
        "gated": bool(reward.get("gated", 0)) if reward else None,
        "media": len(index.get("files") or []),
        "media_skipped": len(index.get("skipped") or []),
        "trajectory": (root / "agent" / "trajectory.json").is_file(),
        "error": exc.get("exception_type") if isinstance(exc, dict) else None,
        "steps": names,
        "step": step,
        "step_rewards": {n: (read_json(trial_dir / "steps" / n / "verifier" / "reward.json") or {}).get("reward")
                         for n in names},
    }


def job_summary(root: Path, job_dir: Path, trials: list[Path]) -> dict:
    result = read_json(job_dir / "result.json") or {}
    summaries = [trial_summary(t) for t in trials]
    finished = result.get("finished_at") or max((s["finished"] or "" for s in summaries), default="") or None
    rel = job_dir.relative_to(root).as_posix()
    return {"job": rel if rel != "." else "", "finished": finished, "n_trials": len(summaries),
            "trials": summaries}


def scan_jobs(root: Path) -> list[dict]:
    """Every directory under root that holds trials, named by its path relative
    to root. Nested layouts (jobs/<family>/<agent>/<model>/<trial>) are found
    too; harbor view only lists the first level. Newest first."""
    jobs: list[dict] = []

    def walk(d: Path, depth: int) -> None:
        if depth > MAX_DEPTH:
            return
        trials = trial_dirs(d)
        if trials:
            jobs.append(job_summary(root, d, trials))
        try:
            subdirs = sorted(p for p in d.iterdir() if p.is_dir() and not is_trial(p)
                             and not p.name.startswith("."))
        except OSError:
            return
        for sub in subdirs:
            walk(sub, depth + 1)

    if root.is_dir():
        walk(root, 0)
    jobs.sort(key=lambda j: j["finished"] or "", reverse=True)
    return jobs


# ---------------------------------------------------------------------------
# one trial
# ---------------------------------------------------------------------------
def list_files(trial_dir: Path) -> list[dict]:
    out = []
    for dirpath, dirnames, filenames in os.walk(trial_dir):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            try:
                out.append({"path": p.relative_to(trial_dir).as_posix(), "bytes": p.stat().st_size})
            except OSError:
                continue
    return out


def trial_detail(trial_dir: Path, step: str | None = None) -> dict:
    detail = trial_summary(trial_dir, step)
    root = step_root(trial_dir, detail["step"]) or trial_dir
    stdout = root / "verifier" / "test-stdout.txt"
    try:
        n_lines = sum(1 for _ in open(stdout, errors="replace"))
    except OSError:
        n_lines = 0
    detail.update(
        reward_json=read_json(root / "verifier" / "reward.json") or {},
        report=read_json(root / "verifier" / "report.json"),
        media_index=read_json(root / "verifier" / "media" / "index.json"),
        files=list_files(trial_dir),
        stdout_lines=n_lines,
    )
    return detail


def read_text(path: Path, tail: int | None = None) -> str | None:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    if tail is not None:
        lines = text.splitlines()
        text = "\n".join(lines[-tail:])
    return text
