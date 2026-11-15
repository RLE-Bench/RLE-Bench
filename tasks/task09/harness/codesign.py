"""Task09 co-design mechanics: instance composition, the probe protocol and
the hardware / software / co-design battery. Verifier-only.

The device is scored as hardware + software:

  hardware  = the submitted lead.xml (springs, masses, counterweights)
  software  = trim.py:
                make_trim(model) -> trim(q) -> tau[nv]    nominal feedforward
                adapt(model, probe) -> params             per-device fitting
                make_trim_adapted(model, params) -> trim(q)
              `model` is always the nominal design model; `probe` is a list
              of (q_target, q_settled) pairs measured on the unknown device
              instance by the probe protocol.

Trim never runs inside the simulation loop. The evaluation samples it at
predetermined configurations (hold targets, path samples, a workspace grid)
in an isolated process and uses those values as a frozen feedforward that
shares the single servo cap with the position servo.

Each seed draws a physical device: per-body mass jitter plus an end-effector
payload at the handle, stratified by seed % 3 over `payload_levels`.
Springs are the design's own and are never perturbed. Software may fit the
instance from `n_probe` static holds run with the feedforward off: the
settled drift encodes the passive residual torque at the settled pose.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import spec as gspec
from .balance import balance_quality, hold_sim, operator_effort
from .probe_protocol import CHOSEN_POSES, TOTAL_MEASUREMENTS
from .scenarios import (GelloEnvelope, compose_lead, fixed_hold_configs,
                        hold_configs, sample_workspace, teleop_path,
                        validate_model_xml, workspace_grid)

CAP = gspec.SERVO_TAU_NM

# --- buildability constraints (see device_buildable) --------------------------
# A tabletop input device: every geom must stay inside a bounded envelope
# around its own body, the printed link structure must span joint-to-joint
# (mass concentrated at the joint axis is a massless connector, not a
# printable part), and the handle must be a real grabbable part.
GEOM_ENVELOPE_M = 0.30      # |geom offset| + bounding radius, per geom
LINK_SPAN_FRAC = 0.85       # printed geoms must reach this fraction of the
                            # way from the link's joint to its child's frame
HANDLE_MASS_MIN = 0.04      # kg, printed handle + encoder insert
STRUCTURE_GAP_TOL_M = 0.003  # assembly tolerance between printed pieces


@dataclass
class CodesignEnvelope(GelloEnvelope):
    """Held-out instance envelope for the co-design battery."""
    payload_max: float = 0.08          # kg, at the handle CoM
    payload_levels: tuple = (0.25, 0.60, 0.95)   # stratified by seed % 3
    n_probe: int = TOTAL_MEASUREMENTS   # probe-protocol hold budget
    probe_noise_std: float = 0.005    # rad, encoder noise on settled reads
    probe_hold_s: float = 1.2
    poke_force_n: float = 2.0          # push-recovery lateral force
    poke_window_s: tuple = (0.5, 0.8)
    poke_end_s: float = 1.8
    grid_stride: int = 9               # workspace-grid subsample for trim/
                                       # headroom sampling


def payload_for_seed(seed: int, env: CodesignEnvelope) -> float:
    return env.payload_levels[int(seed) % len(env.payload_levels)] * env.payload_max


def compose_instance(lead_xml_path: str, seed: int,
                     env: CodesignEnvelope | None = None,
                     perturb: bool = True, payload: bool = True):
    """One device instance: pinned + mass-jittered + (optionally) payloaded.
    Returns (model, data, payload_kg)."""
    env = env or CodesignEnvelope()
    model, data = compose_lead(lead_xml_path, seed=seed, env=env,
                               perturb=perturb)
    pl = 0.0
    if perturb and payload and env.payload_max > 0:
        pl = payload_for_seed(seed, env)
        hid = model.body("lead_handle").id
        model.body_mass[hid] += pl        # at the handle CoM (ipos unchanged)
        mujoco.mj_setConst(model, data)
        mujoco.mj_forward(model, data)
    return model, data, pl


# ---------------------------------------------------------------------------
# Buildability
# ---------------------------------------------------------------------------

def device_buildable(lead_xml_path: str) -> dict:
    """Geometric buildability checks on the raw (uncomposed) model.

    Returns dict(ok=..., problems=[...], n_checks=..., n_failed=...). The
    counts let the checkpoint score the share of buildability checks a device
    passes instead of collapsing on the first problem.
    """
    problems: list = []
    checks = 0
    try:
        validate_model_xml(lead_xml_path)
        model = mujoco.MjModel.from_xml_path(lead_xml_path)
    except Exception as e:
        return dict(ok=False, problems=[f"model does not load: {e}"],
                    n_checks=1, n_failed=1)

    # Inertial boxes retain the original stock motor envelope. Decorative,
    # zero-mass meshes do not add physical constraints or dilute their scores.
    for name, stock in gspec.LEAD_GEOMETRY["stock_geoms"].items():
        if "rbound" in stock:
            try:
                model.geom_rbound[model.geom(name).id] = stock["rbound"]
            except KeyError:
                pass  # stock validation reports the missing motor

    # 1. geom envelope: nothing may stick out on a long boom
    for g in range(model.ngeom):
        if (gspec.LEAD_GEOMETRY["stock_geoms"].get(model.geom(g).name, {}).get("mass") == 0):
            continue
        checks += 1
        reach = float(np.linalg.norm(model.geom_pos[g]) + model.geom_rbound[g])
        if reach > GEOM_ENVELOPE_M:
            name = model.geom(g).name or f"geom{g}"
            problems.append(
                f"{name}: extends {reach:.3f} m from its body frame "
                f"(> {GEOM_ENVELOPE_M} m envelope)")

    # 2. printed link structure must span joint -> child frame
    for i in range(1, gspec.N_JOINTS + 1):
        try:
            b = model.body(f"lead_link{i}")
        except KeyError:
            continue        # structure check reports the missing body
        children = [c for c in range(model.nbody)
                    if model.body_parentid[c] == b.id]
        if not children:
            continue
        child_pos = model.body_pos[children[0]]
        L = float(np.linalg.norm(child_pos))
        if L < 0.02:
            continue        # coincident frames: nothing to span
        u = child_pos / L
        span = 0.0
        printed = []
        assembly = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != b.id:
                continue
            name = model.geom(g).name or ""
            if name.startswith(("printed_", "servo")):
                assembly.append(g)
            if not name.startswith("printed_"):
                continue
            printed.append(g)
            span = max(span, float(np.dot(model.geom_pos[g], u)
                                   + model.geom_rbound[g]))
        checks += 1
        if span < LINK_SPAN_FRAC * L:
            problems.append(
                f"lead_link{i}: printed structure reaches {span:.3f} m of "
                f"the {L:.3f} m to its child frame (< {LINK_SPAN_FRAC:.0%}) "
                f"— a massless connector is not printable")
        centers = {g: model.geom_pos[g].copy() for g in assembly}
        radii = {g: model.geom_rbound[g] for g in assembly}
        # A reversed servo (UR5 J2) carries its case on the child. Include
        # that mounting solid in the graph, expressed in the parent frame.
        try:
            motor = model.geom(f"servo{i+1}").id
            child = children[0]
            if model.geom_bodyid[motor] == child:
                rotation = np.zeros(9)
                mujoco.mju_quat2Mat(rotation, model.body_quat[child])
                centers[motor] = child_pos + rotation.reshape(3, 3) @ model.geom_pos[motor]
                radii[motor] = model.geom_rbound[motor]
                assembly.append(motor)
        except KeyError:
            pass
        # Bounding solids form a simple assembly graph. A real printed path
        # must start at this joint, pass through touching pieces, and reach the
        # child frame; separate endpoint decorations do not make a connection.
        connected = {
            g for g in assembly
            if np.linalg.norm(centers[g])
            <= radii[g] + STRUCTURE_GAP_TOL_M
        }
        frontier = list(connected)
        while frontier:
            source = frontier.pop()
            for target in assembly:
                if target in connected:
                    continue
                center_distance = np.linalg.norm(
                    centers[source] - centers[target])
                touch_distance = (radii[source]
                                  + radii[target]
                                  + STRUCTURE_GAP_TOL_M)
                if center_distance <= touch_distance:
                    connected.add(target)
                    frontier.append(target)
        reaches_child = any(
            np.linalg.norm(centers[g] - child_pos)
            <= radii[g] + STRUCTURE_GAP_TOL_M for g in connected)
        checks += 1
        if not reaches_child:
            problems.append(
                f"lead_link{i}: printed structure is not continuous from "
                f"its joint to the child frame")

    # 3. the handle is a real part
    checks += 1
    try:
        hm = float(model.body_subtreemass[model.body("lead_handle").id])
        if hm < HANDLE_MASS_MIN - 1e-9:
            problems.append(f"lead_handle mass {hm:.3f} kg "
                            f"< {HANDLE_MASS_MIN} kg")
    except KeyError:
        problems.append("lead_handle body missing")

    return dict(ok=not problems, problems=problems,
                n_checks=max(checks, 1), n_failed=len(problems))


# ---------------------------------------------------------------------------
# Probe protocol (feedforward off; the software's only view of the instance)
# ---------------------------------------------------------------------------

def probe_configs(seed: int, env: CodesignEnvelope) -> np.ndarray:
    rng = np.random.default_rng([int(seed), 11])
    return np.vstack([fixed_hold_configs(),
                      sample_workspace(rng, env.n_probe - 3)])


def probe_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Fresh joint-space bounds for probe selection and validation."""
    home = np.asarray(gspec.LEAD_HOME)
    halfwidth = np.asarray(gspec.WORKSPACE_HALFWIDTH)
    return home - halfwidth, home + halfwidth


def validate_probe_plan(plan, model, env: CodesignEnvelope) -> np.ndarray:
    """Validate the whole batch before spending any of the 15 chosen probes."""
    q = np.asarray(plan, dtype=float)
    if env.n_probe != TOTAL_MEASUREMENTS or q.shape != (CHOSEN_POSES, model.nv):
        raise ValueError(f"plan_probe must return exactly ({CHOSEN_POSES}, {model.nv}) angles")
    if not np.all(np.isfinite(q)):
        raise ValueError("probe angles must be finite")
    lower, upper = probe_bounds()
    if np.any(q < lower) or np.any(q > upper):
        raise ValueError("probe angles outside the task workspace")
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            values = q[:, model.jnt_qposadr[j]]
            if np.any(values < model.jnt_range[j, 0]) or np.any(values > model.jnt_range[j, 1]):
                raise ValueError("probe angles outside joint limits")
    return q.copy()


def run_probe(model, data, seed: int, env: CodesignEnvelope,
              configs=None, noise_offset: int = 0) -> list:
    """[(q_target, q_settled_noisy)] on the given instance."""
    configs = probe_configs(seed, env) if configs is None else np.asarray(configs, dtype=float)
    if noise_offset < 0 or noise_offset + len(configs) > env.n_probe:
        raise ValueError("probe budget exceeded")
    noise = np.random.default_rng([int(seed), 10]).normal(
        0.0, env.probe_noise_std, size=(env.n_probe, model.nv))[noise_offset:noise_offset + len(configs)]
    pairs = []
    for i, qt in enumerate(configs):
        hold_sim(model, data, qt, hold_time=env.probe_hold_s)
        qs = np.asarray(data.qpos, dtype=float).copy() + noise[i]
        pairs.append((qt.copy(), qs))
    return pairs


# ---------------------------------------------------------------------------
# Trim sampling contract: the query matrices the evaluation asks trim at
# ---------------------------------------------------------------------------

def qmat_hold(seed: int, env: CodesignEnvelope) -> np.ndarray:
    return hold_configs(seed, env)


def qmat_path(seed: int, env: CodesignEnvelope) -> np.ndarray:
    return teleop_path(seed, env, dt=0.1)


def qmat_grid(env: CodesignEnvelope) -> np.ndarray:
    return workspace_grid()[::env.grid_stride]


def clip_ff(ff) -> np.ndarray:
    return np.clip(np.asarray(ff, dtype=float), -CAP, CAP)


# ---------------------------------------------------------------------------
# Battery pieces
# ---------------------------------------------------------------------------

def device_residual(model, data, seed: int, env: CodesignEnvelope) -> float:
    """H1: worst passive residual of the DEVICE (payload-free instance)."""
    rng = np.random.default_rng([int(seed), 3])
    configs = np.vstack([workspace_grid(),
                         sample_workspace(rng, env.n_residual_samples)])
    return float(balance_quality(model, data, configs)["worst"])


def hold_battery(model, data, seed: int, env: CodesignEnvelope,
                 ff_rows: np.ndarray | None = None) -> dict:
    """Hold droop over the seed's hold configs; ff_rows[i] (if given) is the
    frozen feedforward at hold config i."""
    configs = qmat_hold(seed, env)
    worst_droop, worst_drift = 0.0, 0.0
    for i, qt in enumerate(configs):
        ff = None if ff_rows is None else clip_ff(ff_rows[i])
        r = hold_sim(model, data, qt, hold_time=env.hold_time_s,
                     ee_site=gspec.EE_SITE, trim_ff=ff)
        worst_droop = max(worst_droop, r["ee_droop"])
        worst_drift = max(worst_drift, r["max_joint_drift"])
    return dict(max_ee_droop=worst_droop, max_joint_drift=worst_drift)


def effort_battery(model, data, seed: int, env: CodesignEnvelope,
                   ff_mat: np.ndarray | None = None) -> dict:
    """Operator effort along the seed's teleop path with a per-sample frozen
    feedforward."""
    eff = operator_effort(model, data, qmat_path(seed, env), dt=0.1, ff=ff_mat)
    return dict(peak=eff["peak"], rms=eff["rms"])


def headroom(ff_grid: np.ndarray | None) -> float:
    """Static servo headroom left by the feedforward over the grid sample.
    No software => the whole cap is headroom (the passive numbers speak)."""
    if ff_grid is None:
        return CAP
    return float(CAP - np.abs(clip_ff(ff_grid)).max())


def poke_test(model, data, q_target, ff_row=None, env: CodesignEnvelope | None = None) -> dict:
    """Push-recovery: hold q_target with the frozen feedforward, shove the
    handle sideways during the window, measure peak/final EE deviation. The
    feedforward and servo share one torque cap: software that spends the cap
    statically has nothing left to fight the push."""
    env = env or CodesignEnvelope()
    q_target = np.asarray(q_target, dtype=float)
    ff = np.zeros(model.nv) if ff_row is None else clip_ff(ff_row)
    hid = model.body("lead_handle").id
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q_target
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    ee0 = data.site(gspec.EE_SITE).xpos.copy()
    peak = 0.0
    t0, t1 = env.poke_window_s
    for s in range(int(env.poke_end_s / model.opt.timestep)):
        t = s * model.opt.timestep
        tau = (gspec.SERVO_KP * (q_target - data.qpos)
               - gspec.SERVO_KD * data.qvel + ff)
        data.qfrc_applied[:] = np.clip(tau, -CAP, CAP)
        data.xfrc_applied[hid, :3] = ((env.poke_force_n, 0.0, 0.0)
                                      if t0 <= t < t1 else 0.0)
        mujoco.mj_step(model, data)
        dev = float(np.linalg.norm(data.site(gspec.EE_SITE).xpos - ee0))
        peak = max(peak, dev)
    data.xfrc_applied[hid, :3] = 0.0
    return dict(peak=peak,
                final=float(np.linalg.norm(data.site(gspec.EE_SITE).xpos - ee0)))


# ---------------------------------------------------------------------------
# Battery orchestration
# ---------------------------------------------------------------------------
# A trim provider supplies frozen feedforward samples:
#     provider(kind, seed, qmat) -> (len(qmat), nv) array or None
# kind in {"hold", "path", "grid"}. None means no software (passive run).
# The scorer builds providers from isolated-process samples of the submitted
# trim.py; local_provider() builds the same thing in-process for calibration.

def codesign_battery(lead_xml_path: str, seeds, env: CodesignEnvelope | None = None,
                     provider=None) -> dict:
    """The full H/S/B battery on one hardware design + one software provider.
    Returns worst-case aggregates plus per-seed detail."""
    env = env or CodesignEnvelope()
    grid_q = qmat_grid(env)
    agg = dict(device_residual=0.0, passive_droop=0.0, passive_effort=0.0,
               sw_droop=0.0, sw_effort=0.0, headroom=CAP,
               poke_peak=0.0, poke_final=0.0, payloads=[], per_seed=[])
    for seed in seeds:
        # H1 on the payload-free instance
        m0, d0, _ = compose_instance(lead_xml_path, seed, env, payload=False)
        h1 = device_residual(m0, d0, seed, env)
        # the operating instance (jitter + payload)
        model, data, pl = compose_instance(lead_xml_path, seed, env)
        hp = hold_battery(model, data, seed, env, ff_rows=None)
        ep = effort_battery(model, data, seed, env, ff_mat=None)
        ff_hold = ff_path = ff_grid = None
        if provider is not None:
            ff_hold = provider("hold", seed, qmat_hold(seed, env))
            ff_path = provider("path", seed, qmat_path(seed, env))
            ff_grid = provider("grid", seed, grid_q)
        hs = hold_battery(model, data, seed, env, ff_rows=ff_hold)
        es = effort_battery(model, data, seed, env, ff_mat=ff_path)
        hr = headroom(ff_grid)
        # push-recovery at the fixed corners + the hungriest grid pose
        poke_sets = [(fixed_hold_configs()[i],
                      None if ff_hold is None else ff_hold[i])
                     for i in range(3)]
        if ff_grid is not None:
            i_hungry = int(np.abs(clip_ff(ff_grid)).max(axis=1).argmax())
            poke_sets.append((grid_q[i_hungry], ff_grid[i_hungry]))
        pk = pf = 0.0
        for qt, ff_row in poke_sets:
            p = poke_test(model, data, qt, ff_row=ff_row, env=env)
            pk = max(pk, p["peak"])
            pf = max(pf, p["final"])
        seed_row = dict(seed=seed, payload=pl, device_residual=h1,
                        passive_droop=hp["max_ee_droop"],
                        passive_effort=ep["peak"],
                        sw_droop=hs["max_ee_droop"], sw_effort=es["peak"],
                        headroom=hr, poke_peak=pk, poke_final=pf)
        agg["per_seed"].append(seed_row)
        agg["payloads"].append(pl)
        agg["device_residual"] = max(agg["device_residual"], h1)
        agg["passive_droop"] = max(agg["passive_droop"], hp["max_ee_droop"])
        agg["passive_effort"] = max(agg["passive_effort"], ep["peak"])
        agg["sw_droop"] = max(agg["sw_droop"], hs["max_ee_droop"])
        agg["sw_effort"] = max(agg["sw_effort"], es["peak"])
        agg["headroom"] = min(agg["headroom"], hr)
        agg["poke_peak"] = max(agg["poke_peak"], pk)
        agg["poke_final"] = max(agg["poke_final"], pf)
    return agg


def local_provider(lead_xml_path: str, trim_module, seeds,
                   env: CodesignEnvelope | None = None, adapted: bool = True):
    """Build a provider by running the probe protocol and the given trim
    module in-process, over the same query matrices the scorer samples."""
    env = env or CodesignEnvelope()
    nominal_model, _ = compose_lead(lead_xml_path, seed=0, perturb=False)
    trims = {}
    for seed in seeds:
        if adapted:
            model, data, _ = compose_instance(lead_xml_path, seed, env)
            sample = run_probe(model, data, seed, env, [gspec.LEAD_HOME])[0]
            lower, upper = probe_bounds()
            plan = validate_probe_plan(trim_module.plan_probe(
                nominal_model, tuple(q.copy() for q in sample), lower, upper), model, env)
            pairs = [sample] + run_probe(model, data, seed, env, plan, noise_offset=1)
            params = trim_module.adapt(nominal_model, pairs)
            trims[seed] = trim_module.make_trim_adapted(nominal_model, params)
        else:
            trims[seed] = trim_module.make_trim(nominal_model)

    def provider(kind, seed, qmat):
        fn = trims[seed]
        return np.vstack([clip_ff(fn(np.asarray(q, dtype=float).copy()))
                          for q in qmat])
    return provider
