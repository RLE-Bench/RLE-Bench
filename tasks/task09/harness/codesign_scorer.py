"""Task09 submission scoring: orchestration + weighted aggregate.

score_submission() is the single verifier entry point. Every submission file
is untrusted: physical quantities come from re-running the harness battery on
the submitted model; trim.py executes only in the sandbox subprocess, sampled
on predetermined query matrices (the frozen-feedforward contract in
codesign.py).

Expected submission layout:
    lead.xml    the co-designed hardware (springs + masses + counterweights)
    trim.py     make_trim(model); adapt(model, probe); make_trim_adapted(...)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import codesign_variant_config as ccfg
from . import spec as gspec
from .codesign import (CodesignEnvelope, codesign_battery, compose_instance,
                       device_buildable, hold_battery, effort_battery,
                       qmat_grid, qmat_hold, qmat_path, run_probe, validate_probe_plan,
                       probe_bounds)
from .codesign_checkpoints import aggregate_codesign, evaluate_codesign
from .sandbox import run_jobs
from .scenarios import compose_lead, validate_model_xml
from .validity import check_validity

SANDBOX_TIMEOUT_S = 300.0


@dataclass
class Submission:
    lead_xml: str | None = None
    has_trim: bool = False           # make_trim present
    has_adapt: bool = False          # adapt + make_trim_adapted present
    errors: list = field(default_factory=list)


def load_submission(submission_dir: str) -> Submission:
    """Record which files are present without importing agent Python."""
    sub = Submission()
    lead = os.path.join(submission_dir, "lead.xml")
    if os.path.exists(lead):
        sub.lead_xml = lead
    else:
        sub.errors.append("lead.xml missing")
    trim_py = os.path.join(submission_dir, "trim.py")
    if os.path.exists(trim_py):
        src = open(trim_py, encoding="utf-8", errors="replace").read()
        sub.has_trim = "def make_trim" in src
        sub.has_adapt = ("def adapt" in src
                         and "def make_trim_adapted" in src)
        if not sub.has_trim:
            sub.errors.append("trim.py has no make_trim()")
    else:
        sub.errors.append("trim.py missing (software is required)")
    return sub


def _springs_legal(lead_xml: str) -> float:
    """Share of joints whose spring is physical hardware.

    A negative stiffness pushes away from its reference: that is a
    simulation-only 'spring'. Returns the share of legal joints, so one bad
    spring out of seven costs a seventh of the checkpoint rather than all of
    it. 0.0 when the model cannot be read at all.
    """
    try:
        validate_model_xml(lead_xml)
        model = mujoco.MjModel.from_xml_path(lead_xml)
    except Exception:
        return 0.0
    stiffness = np.asarray(model.jnt_stiffness, dtype=float)
    if stiffness.size == 0:
        return 1.0        # no springs is a legal (if unbalanced) design
    return float(np.mean(stiffness >= 0.0))


def _sandbox_for_arm(submission_dir: str, arm_xml: str, sub: Submission,
                     env: CodesignEnvelope, seeds, errors: list,
                     want_grid: bool = True):
    """Run probes on the arm's hidden instances, then sample the submitted
    software (nominal + adapted) on the arm's query matrices."""
    jobs: dict = {}
    probe_log = {}
    grid_q = qmat_grid(env)
    if sub.has_trim:
        jobs["ctrim"] = {}
        for seed in seeds:
            jobs["ctrim"][f"hold::{seed}"] = qmat_hold(seed, env)
            jobs["ctrim"][f"path::{seed}"] = qmat_path(seed, env)
        if want_grid:
            jobs["ctrim"]["grid"] = grid_q
    if sub.has_adapt:
        jobs["cadapt"] = {}
        planning = {}
        instances = {}
        lower, upper = probe_bounds()
        for seed in seeds:
            model, data, _ = compose_instance(arm_xml, seed, env)
            if model.nv != gspec.N_JOINTS:
                errors.append(f"instance {seed}: wrong dof count, probes skipped")
                continue
            sample = run_probe(model, data, seed, env, [gspec.LEAD_HOME])[0]
            probe_log[str(seed)] = dict(measurements=1, targets=[sample[0].tolist()])
            instances[str(seed)] = model, data, sample
            planning[str(seed)] = dict(sample=sample, lower=lower, upper=upper)
        planned = run_jobs(submission_dir, arm_xml, {"plan_probe": planning},
                           timeout=SANDBOX_TIMEOUT_S) or {}
        for key, (model, data, sample) in instances.items():
            seed = int(key)
            response = planned.get("plan_probe", {}).get(key) or {}
            try:
                if "error" in response:
                    raise ValueError(response["error"])
                configs = validate_probe_plan(response.get("configs"), model, env)
            except (ValueError, TypeError) as exc:
                probe_log[key]["error"] = str(exc)
                errors.append(f"instance {seed}: probe plan rejected: {exc}")
                continue
            probe = [sample] + run_probe(model, data, seed, env, configs, noise_offset=1)
            probe_log[key] = dict(measurements=len(probe), targets=[q.tolist() for q, _ in probe])
            qmats = {f"hold::{seed}": qmat_hold(seed, env),
                     f"path::{seed}": qmat_path(seed, env)}
            if want_grid:
                qmats["grid"] = grid_q
            jobs["cadapt"][str(seed)] = dict(probe=probe, qmats=qmats)
    if not jobs:
        return {}
    out = run_jobs(submission_dir, arm_xml, jobs, timeout=SANDBOX_TIMEOUT_S)
    if out is None:
        errors.append("sandbox failed or timed out on the software battery")
        return {"probe_report": probe_log}
    out["probe_report"] = probe_log
    return out


def _provider_from(sandbox: dict, seeds):
    """Best-available provider: adapted per instance, else nominal, else None."""
    ctrim = sandbox.get("ctrim") or {}
    cadapt = sandbox.get("cadapt") or {}

    def lookup(kind, seed, qmat):
        key = "grid" if kind == "grid" else f"{kind}::{seed}"
        per = cadapt.get(str(seed))
        if per is not None and per.get(key) is not None:
            return np.asarray(per[key], dtype=float)
        if key in ctrim:
            return np.asarray(ctrim[key], dtype=float)
        return None

    if not ctrim and not any(v for v in cadapt.values()):
        return None
    return lookup


def score_submission(submission_dir: str, sref_xml: str | None = None,
                     env: CodesignEnvelope | None = None,
                     variant: str | None = None) -> dict:
    """Full verifier pipeline: aggregate() report + battery stats."""
    if variant is not None and variant != gspec.VARIANT_NAME:
        raise ValueError(f"verifier process is bound to {gspec.VARIANT_NAME!r}, not {variant!r}")
    mujoco.set_mju_user_warning(lambda m: None)   # bad models slam limits
    env = env or CodesignEnvelope()
    seeds = ccfg.HIDDEN_SEEDS
    if sref_xml is None:
        from .codesign_oracle import CODESIGN_MODELS
        sref_xml = os.path.join(CODESIGN_MODELS, gspec.VARIANT_NAME, "lead_sref.xml")

    sub = load_submission(submission_dir)
    if sub.lead_xml is None:
        report = aggregate_codesign({})
        report["errors"] = sub.errors
        return report

    # validity describes the submitted design: composed without the per-seed
    # tolerance perturbation
    validity = check_validity(
        sub.lead_xml,
        compose=lambda p, seed=0: compose_lead(p, seed=seed, env=env,
                                              perturb=False))
    springs_share = _springs_legal(sub.lead_xml)
    buildable = device_buildable(sub.lead_xml)
    if not buildable["ok"]:
        sub.errors.extend(buildable["problems"])

    # --- the agent's own arm: battery with its best software ------------------
    batt = {}
    probe_reports = {}
    if validity.get("loads") and validity.get("structure", {}).get("ok"):
        own_sandbox = _sandbox_for_arm(submission_dir, sub.lead_xml, sub, env,
                                       seeds, sub.errors)
        probe_reports["submitted_arm"] = own_sandbox.get("probe_report", {})
        provider = _provider_from(own_sandbox, seeds)
        batt = codesign_battery(sub.lead_xml, seeds, env, provider=provider)
    else:
        sub.errors.append("model invalid: battery skipped")

    # --- software tier on the harness reference arm ---------------------------
    s_tier = dict(nominal_droop=None, adapted_droop=None, adapted_effort=None)
    if sub.has_trim and os.path.exists(sref_xml):
        sref_sandbox = _sandbox_for_arm(submission_dir, sref_xml, sub, env,
                                        seeds, sub.errors, want_grid=False)
        probe_reports["reference_arm"] = sref_sandbox.get("probe_report", {})
        ctrim = sref_sandbox.get("ctrim") or {}
        cadapt = sref_sandbox.get("cadapt") or {}
        nom_droop = ada_droop = ada_eff = 0.0
        nom_ok = ada_ok = True
        for seed in seeds:
            model, data, _ = compose_instance(sref_xml, seed, env)
            hk, pk = f"hold::{seed}", f"path::{seed}"
            if hk in ctrim:
                nom_droop = max(nom_droop, hold_battery(
                    model, data, seed, env, ff_rows=ctrim[hk])["max_ee_droop"])
            else:
                nom_ok = False
            per = cadapt.get(str(seed))
            if per is not None and per.get(hk) is not None:
                ada_droop = max(ada_droop, hold_battery(
                    model, data, seed, env, ff_rows=per[hk])["max_ee_droop"])
                ada_eff = max(ada_eff, effort_battery(
                    model, data, seed, env, ff_mat=per[pk])["peak"])
            else:
                ada_ok = False
        if nom_ok:
            s_tier["nominal_droop"] = nom_droop
        if ada_ok:
            s_tier["adapted_droop"] = ada_droop
            s_tier["adapted_effort"] = ada_eff

    scores = evaluate_codesign(validity, springs_share, batt, s_tier,
                               buildable=buildable)
    report = aggregate_codesign(scores)
    report["errors"] = sub.errors
    report["motion_clearance"] = validity.get("motion_clearance", {})
    report["probe_protocol"] = probe_reports
    if batt:
        report["battery"] = {k: (round(v, 5) if isinstance(v, float) else v)
                             for k, v in batt.items() if k != "per_seed"}
        report["battery_per_seed"] = [
            {k: (round(v, 5) if isinstance(v, float) else v)
             for k, v in row.items()} for row in batt["per_seed"]]
    report["s_tier"] = {k: (None if v is None else round(v, 5))
                        for k, v in s_tier.items()}
    return report
