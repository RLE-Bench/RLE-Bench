"""Verifier entry point for task07 (verifier-only).

score_submission(submission_dir) runs the untrusted policy package through
the sandbox on the hidden evaluation seeds — one fresh subprocess per
episode — and scores harness instrumentation only; nothing the agent
self-reports is used.

Reward, per episode:
    PERFECT_BONUS*perfect + W_CLEAR*clear_curve(f)
    + W_SPEED*min(tp/TP_CAP, 1) + W_SPEED_PERFECT*perfect*min(tp/TP_BONUS, 1)
    - W_FLOOR*floor_drops - W_DAMAGE*damage_events, clamped to [0,1],
where perfect = bin emptied with zero floor drops, clear_curve pays the
hard tail (cubic past the knee), and tp is parts/min paced against the
full budget unless the bin was emptied. Mean over episodes.

Gate: the policy loads, answers the contract on a smoke episode with few
faults, the sim completes, and a repeated smoke run is bit-deterministic.
A failed gate caps the reward at GATE_CAP.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

from rlebench.core.media import VIDEO_FPS

from . import config, runtime, sandbox, spec

_SEEDS_PATH = os.path.join(os.path.dirname(__file__), "eval_seeds.json")

# Verifier-only key: the policy receives a deterministic RNG seed without
# learning the world seed that reproduces the hidden pile and sensor streams.
_POLICY_SEED_KEY = bytes.fromhex(
    "cf8e3e827459807287042ac1e80ba336"
    "2eb987026d66d1851e5c52de6623655f"
)


def _policy_seed(world_seed: int) -> int:
    message = str(int(world_seed)).encode("ascii")
    digest = hmac.new(_POLICY_SEED_KEY, message, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _load_seeds(seeds=None):
    if seeds is not None:
        return int(seeds["smoke"]), list(seeds["eval"])
    with open(_SEEDS_PATH) as f:
        d = json.load(f)
    return int(d["smoke"]), list(d["eval"])


EVAL_VIDEO_EVERY = round(spec.CTRL_HZ / VIDEO_FPS)   # 20 Hz control -> 10 fps video


def _run_sandboxed(submission_dir: str, seed: int, budget_t: float,
                   wall_budget_s: float | None = None, video=None):
    episode_start = time.monotonic()
    with sandbox.PolicyProcess(submission_dir) as p:
        p.check_loaded()
        log = runtime.run_episode(p, seed, budget_t=budget_t, render=True,
                                  policy_seed=_policy_seed(seed),
                                  wall_budget_s=wall_budget_s,
                                  video=video, video_mode="camera",
                                  video_every=EVAL_VIDEO_EVERY,
                                  video_size=(640, 360))
        policy_wall_s = p.policy_wall_s
        diagnostics = {
            "reset_wall_s": round(p.reset_wall_s, 6),
            "act_wall_s": round(p.act_wall_s, 6),
            "policy_wall_s": round(policy_wall_s, 6),
            "n_acts": p.n_acts,
            "sandbox_dead": p.dead,
            "sandbox_stderr_tail": p.stderr_tail(2000),
        }
    episode_wall_s = time.monotonic() - episode_start
    diagnostics.update(
        episode_wall_s=round(episode_wall_s, 6),
        harness_wall_s=round(max(episode_wall_s - policy_wall_s, 0.0), 6),
    )
    return log, diagnostics


def _smoke_diagnostics(log: runtime.EpisodeLog, diagnostics: dict) -> dict:
    """Safe, compact gate diagnostics without hidden scene state."""
    return {
        "ticks": log.ticks,
        "policy_faults": log.policy_faults,
        "fault_reasons": list(log.fault_reasons),
        "aborted": log.aborted,
        **diagnostics,
    }


def clear_curve(f: float) -> float:
    """Progressive clearance credit: the first CLEAR_KNEE of the bin earns
    only CLEAR_BASE of the credit (those parts are easy); the remainder
    rises as a degree-CLEAR_P curve — the hard tail is where the credit is."""
    knee, base = config.CLEAR_KNEE, config.CLEAR_BASE
    if f <= knee:
        return base * (f / knee)
    return base + (1.0 - base) * ((f - knee) / (1.0 - knee)) ** config.CLEAR_P


def episode_throughput(m: dict, budget_t: float = spec.EPISODE_T) -> float:
    """Parts/min. Only a fully emptied bin gets the makespan denominator —
    an unfinished episode is paced against the whole budget, so stalling
    early cannot inflate the rate."""
    finished = m["parts_cleared"] == m["n_parts"] and m["n_parts"] > 0
    denom = m["makespan_s"] if finished and m["makespan_s"] else budget_t
    return 60.0 * m["parts_cleared"] / denom if denom else 0.0


def episode_score(m: dict, budget_t: float = spec.EPISODE_T) -> float:
    perfect = (m["n_parts"] > 0
               and m["parts_cleared"] == m["n_parts"]
               and m["floor_drops"] == 0)
    tp = episode_throughput(m, budget_t)
    s = (config.PERFECT_BONUS * perfect
         + config.W_CLEAR * clear_curve(m["clear_frac"])
         + config.W_SPEED * min(tp / config.TP_CAP_PPM, 1.0)
         + config.W_SPEED_PERFECT * perfect
         * min(tp / config.TP_BONUS_CAP_PPM, 1.0)
         - config.W_FLOOR * m["floor_drops"]
         - config.W_DAMAGE * m["damage_events"]
         - config.W_BIN_HIT * m.get("bin_hits", 0))
    return max(0.0, min(1.0, s))


def score_submission(submission_dir: str, seeds: dict | None = None,
                     media=None) -> dict:
    """``media``: a rlebench.core.media.Media; each eval episode is recorded
    into it as ep<i>.mp4 (overview camera, 10 fps, 640x360). Recording is a
    pure observer of the episode and is not charged to the submission-only
    EPISODE_WALL_BUDGET_S; without ffmpeg or GL the media index says so."""
    smoke_seed, eval_seeds = _load_seeds(seeds)
    report: dict = {"errors": [], "episodes": []}

    if not os.path.exists(os.path.join(submission_dir, "policy.py")):
        report["errors"].append("policy.py missing")
        report.update(reward=0.0, raw_total=0.0, gated=True,
                      gate={"load": False, "contract": False,
                            "stable": False, "determinism": False})
        return report

    # --- gate: smoke episode twice (contract + determinism) ----------------
    # Smoke gets the same cumulative submission allowance as scored episodes;
    # without it, a policy returning just below the per-act hang cap could
    # consume the verifier's entire outer timeout before evaluation starts.
    try:
        smoke1, diag1 = _run_sandboxed(
            submission_dir, smoke_seed, config.SMOKE_T,
            wall_budget_s=spec.EPISODE_WALL_BUDGET_S)
    except sandbox.PolicyLoadError as exc:
        report["errors"].append(str(exc))
        report.update(reward=0.0, raw_total=0.0, gated=True,
                      gate={"load": False, "contract": False,
                            "stable": False, "determinism": False})
        return report
    if smoke1.aborted == "policy_startup":
        report["smoke"] = [_smoke_diagnostics(smoke1, diag1)]
        report["errors"].append(f"policy reset failed: {diag1['sandbox_dead']} :: "
                                f"{diag1['sandbox_stderr_tail'][-500:]}")
        report.update(reward=0.0, raw_total=0.0, gated=True,
                      gate={"load": False, "contract": False,
                            "stable": False, "determinism": False})
        return report
    if smoke1.ticks > 0 and smoke1.policy_faults >= smoke1.ticks:
        report["smoke"] = [_smoke_diagnostics(smoke1, diag1)]
        detail = diag1["sandbox_dead"] or "faulted on every smoke step"
        report["errors"].append(f"smoke 0: {detail} :: "
                                f"{diag1['sandbox_stderr_tail'][-500:]}")
        report.update(reward=0.0, raw_total=0.0, gated=True,
                      gate={"load": diag1["sandbox_dead"] is None,
                            "contract": False,
                            "stable": smoke1.aborted is None,
                            "determinism": False})
        return report
    smoke2, diag2 = _run_sandboxed(
        submission_dir, smoke_seed, config.SMOKE_T,
        wall_budget_s=spec.EPISODE_WALL_BUDGET_S)
    report["smoke"] = [_smoke_diagnostics(smoke1, diag1),
                       _smoke_diagnostics(smoke2, diag2)]
    dead1 = diag1["sandbox_dead"]
    fault_frac = smoke1.policy_faults / max(smoke1.ticks, 1)
    gate = {
        "load": dead1 is None or smoke1.policy_faults < smoke1.ticks,
        "contract": fault_frac <= 0.10,
        "stable": smoke1.aborted is None,
        "determinism": smoke1.digest == smoke2.digest,
    }
    gate_ok = all(gate.values())
    for i, (smoke, diag) in enumerate(((smoke1, diag1), (smoke2, diag2))):
        dead, err = diag["sandbox_dead"], diag["sandbox_stderr_tail"]
        if dead is not None:
            report["errors"].append(f"smoke {i}: {dead} :: {err[-500:]}")
        elif smoke.policy_faults and err.strip():
            report["errors"].append(f"smoke {i} policy stderr: {err[-500:]}")
    if not gate["determinism"]:
        report["errors"].append(
            "smoke episodes diverged: policy must be deterministic "
            "(seeded RNG only, no wall clock)")

    # --- scored battery -----------------------------------------------------
    scores = []
    for i, seed in enumerate(eval_seeds):
        video = media.video(f"ep{i}.mp4") if media is not None else None
        log, diagnostics = _run_sandboxed(
            submission_dir, seed, spec.EPISODE_T,
            wall_budget_s=spec.EPISODE_WALL_BUDGET_S, video=video)
        if media is not None:
            media.finish(video)
        m = log.metrics()
        # Detailed event poses are public-design-seed dev diagnostics. Do not
        # leak hidden evaluation scene state through the verifier report.
        m.pop("penalty_events", None)
        m["episode_tp_ppm"] = round(episode_throughput(m), 2)
        m["episode_score"] = round(episode_score(m), 4)
        m.update(diagnostics)
        del m["clear_times"]
        report["episodes"].append(m)
        scores.append(m["episode_score"])
        dead = diagnostics["sandbox_dead"]
        err = diagnostics["sandbox_stderr_tail"]
        if dead is not None:
            report["errors"].append(f"eval {i}: {dead} :: {err[-500:]}")
        elif log.policy_faults and err.strip():
            report["errors"].append(f"eval {i} policy stderr: {err[-500:]}")

    total = sum(scores) / len(scores) if scores else 0.0
    reward = min(total, config.GATE_CAP) if not gate_ok else total
    eps = report["episodes"]

    def mean(k):
        vals = [e[k] for e in eps if e[k] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    report.update(
        reward=round(reward, 4),
        raw_total=round(total, 4),
        gated=not gate_ok,
        gate=gate,
        clear_frac=mean("clear_frac"),
        parts_cleared=sum(e["parts_cleared"] for e in eps),
        n_parts=sum(e["n_parts"] for e in eps),
        speed_merit=mean("speed_merit"),
        throughput_ppm=mean("throughput_ppm"),
        makespan_s=mean("makespan_s"),
        floor_drops=sum(e["floor_drops"] for e in eps),
        damage_events=sum(e["damage_events"] for e in eps),
        bin_hits=sum(e.get("bin_hits", 0) for e in eps),
        policy_faults=sum(e["policy_faults"] for e in eps),
        episode_wall_s_total=round(sum(e["episode_wall_s"] for e in eps), 4),
        policy_wall_s_total=round(sum(e["policy_wall_s"] for e in eps), 4),
        harness_wall_s_total=round(sum(e["harness_wall_s"] for e in eps), 4),
    )
    return report
