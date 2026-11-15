"""Stage-1 validity checks for submitted lead-arm models.

Never raises on a bad model: an unloadable submission is an expected
verifier input.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from . import spec as gspec
from .balance import hold_sim
from .kinematics import structure_check
from .scenarios import validate_model_xml
from rlebench.core.model import check_inertias_valid

SETTLE_SPEED_MAX = 0.1     # rad/s at the end of a 1 s hold: "no blowup"


def _geom_volume(g) -> float | None:
    gtype = g.get("type", "sphere")
    size = [float(v) for v in g.get("size", "0").split()]
    if gtype == "sphere":
        return 4.0 / 3.0 * np.pi * size[0] ** 3
    if gtype == "cylinder":
        return np.pi * size[0] ** 2 * (2.0 * size[1])
    if gtype == "box":
        return 8.0 * size[0] * size[1] * size[2]
    if gtype == "capsule":
        if g.get("fromto"):
            ft = [float(v) for v in g.get("fromto").split()]
            length = float(np.linalg.norm(np.array(ft[3:]) - np.array(ft[:3])))
        else:
            length = 2.0 * size[1]
        return np.pi * size[0] ** 2 * length + 4.0 / 3.0 * np.pi * size[0] ** 3
    return None   # mesh/hfield/plane: not a printable device part


def density_report(lead_xml_path: str) -> dict:
    """Printability of supplied parts; additions use the model validity checks.

    Supplied meshes remain immutable. Added meshes and geoms do not
    contribute to the density score, including additions named printed_*.
    """
    problems, per_geom = [], {}
    worst = 0.0        # worst violation ratio over all geoms (<= 1: fine)
    try:
        root = validate_model_xml(lead_xml_path)
    except Exception as e:
        return dict(ok=False, problems=[f"unparseable XML: {e}"], per_geom={},
                    worst_ratio=float("inf"))
    mesh_volumes = {}
    supplied_geoms = gspec.LEAD_GEOMETRY["supplied_geoms"]
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    for name, expected_digest in gspec.LEAD_GEOMETRY["supplied_meshes"].items():
        try:
            mesh = root.find(f"./asset/mesh[@name='{name}']")
            if mesh is None:
                raise ValueError("supplied mesh missing")
            path = Path(lead_xml_path).parent / meshdir / mesh.get("file", "")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_digest:
                raise ValueError("stock mesh bytes changed")
            volume = gspec.LEAD_GEOMETRY["mesh_volumes"][digest]
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            if not np.allclose(scale, 0.001, rtol=0, atol=1e-12):
                raise ValueError("stock mesh scale changed")
            if mesh.get("inertia") != "legacy":
                raise ValueError("stock mesh inertia must be legacy")
            mesh_volumes[mesh.get("name")] = volume
        except (OSError, KeyError, ValueError) as exc:
            problems.append(f"unrecognized or altered mesh: {name}: {exc}")
            worst = float("inf")
    for name, mesh_name in supplied_geoms.items():
        geom = root.find(f".//geom[@name='{name}']")
        if geom is None or geom.get("mesh") != mesh_name:
            problems.append(f"{name}: supplied geom missing or mesh changed")
            worst = float("inf")
    # Inertial overrides/default classes would disconnect checked geometry
    # density from the mass used by the simulator.
    geom_defaults = [g for d in root.iter("default") for g in d.findall("geom")
                     if set(g.attrib) - {"contype", "conaffinity", "rgba"}]
    if (list(root.iter("inertial")) or geom_defaults
            or any(g.get("class") for g in root.iter("geom"))
            or any(b.get("childclass") for b in root.iter("body"))):
        problems.append("explicit inertials and physical geom defaults/classes are not allowed")
        worst = float("inf")
    try:
        model = mujoco.MjModel.from_xml_path(lead_xml_path)
        for name, expected in gspec.LEAD_GEOMETRY["stock_geoms"].items():
            geom = root.find(f".//geom[@name='{name}']")
            gid = model.geom(name).id
            if (geom is None or geom.get("type") != expected["type"]
                    or geom.get("mesh") != expected["mesh"]
                    or not np.isclose(float(geom.get("mass", "nan")), expected["mass"], atol=1e-12, rtol=0)
                    or not np.allclose(model.geom_size[gid], expected["size"], atol=1e-12, rtol=0)
                    or model.body(int(model.geom_bodyid[gid])).name != expected["body"]
                    or not np.allclose(model.geom_pos[gid], expected["pos"], atol=1e-8, rtol=0)
                    or abs(np.dot(model.geom_quat[gid], expected["quat"])) < 1-1e-8):
                problems.append(f"{name}: stock motor mass or mounting changed")
                worst = float("inf")
    except (ValueError, KeyError, TypeError):
        # Unloadable additions belong to C1.load, not printability.
        # Missing supplied geoms are reported independently above.
        pass
    for g in root.iter("geom"):
        name = g.get("name", "<unnamed>")
        if name not in supplied_geoms:
            continue
        if g.get("mass") is not None:
            vol = (mesh_volumes.get(g.get("mesh")) if g.get("type") == "mesh"
                   else _geom_volume(g))
            if vol is None or vol <= 0:
                problems.append(f"{name}: cannot derive volume for density check")
                worst = float("inf")
                continue
            rho = float(g.get("mass")) / vol
        else:
            rho = float(g.get("density", "1000"))
        per_geom[name] = rho
        if name.startswith("printed_"):
            worst = max(worst, rho / gspec.DENSITY_PRINTED_MAX,
                        gspec.DENSITY_PRINTED_MIN / rho if rho > 0
                        else float("inf"))
        worst = max(worst, rho / gspec.DENSITY_ANY_MAX)
        if name.startswith("printed_") and rho < gspec.DENSITY_PRINTED_MIN:
            problems.append(f"{name}: density {rho:.0f} kg/m^3 is too low "
                            f"to represent a physical printed part "
                            f"(< {gspec.DENSITY_PRINTED_MIN:.0f})")
        if name.startswith("printed_") and rho > gspec.DENSITY_PRINTED_MAX:
            problems.append(f"{name}: density {rho:.0f} kg/m^3 is not printable "
                            f"(> {gspec.DENSITY_PRINTED_MAX:.0f})")
        if rho > gspec.DENSITY_ANY_MAX:
            problems.append(f"{name}: density {rho:.0f} kg/m^3 exceeds solid "
                            f"metal (> {gspec.DENSITY_ANY_MAX:.0f})")
    return dict(ok=not problems, problems=problems, per_geom=per_geom,
                worst_ratio=float(worst))


def check_validity(lead_xml_path: str, compose=None, seed: int = 0) -> dict:
    """Full stage-1 report. `compose` is scenarios.compose_lead (injected to
    avoid a circular import); falls back to a bare load when not given."""
    out: dict = dict(loads=False)
    out["density"] = density_report(lead_xml_path)
    try:
        if compose is not None:
            model, data = compose(lead_xml_path, seed=seed)
        else:
            validate_model_xml(lead_xml_path)
            model = mujoco.MjModel.from_xml_path(lead_xml_path)
            data = mujoco.MjData(model)
    except Exception as e:
        out["load_error"] = str(e)
        return out
    out["loads"] = True
    out["inertia"] = check_inertias_valid(model)
    out["inertia_ok"] = all(v["ok"] for v in out["inertia"].values())
    out["total_mass"] = float(mujoco.mj_getTotalmass(model))
    out["mass_ok"] = out["total_mass"] <= gspec.MASS_BUDGET_KG
    out["mass_ratio"] = out["total_mass"] / gspec.MASS_BUDGET_KG
    # each moving link must at least carry its servo — a near-massless arm
    # would make balancing trivial and is not a buildable device
    light = []
    link_worst = 0.0
    for name, minimum in gspec.STOCK_BODY_MASS.items():
        try:
            m = float(model.body(name).mass[0])
        except KeyError:
            continue   # structure check reports the missing body
        link_worst = max(link_worst, minimum / m if m > 0
                         else float("inf"))
        if m < minimum - 1e-9:
            light.append(f"{name} mass {m:.3f} kg < stock case mass {minimum}")
    out["link_mass_ok"] = not light
    out["link_mass_problems"] = light
    out["link_mass_worst_ratio"] = float(link_worst)
    out["structure"] = structure_check(model, data)
    # settle requires the contract DoF layout; a wrong-DoF model fails
    # structure and cannot be held
    if model.nq == gspec.N_JOINTS and model.nv == gspec.N_JOINTS:
        try:
            settle = hold_sim(model, data, np.asarray(gspec.LEAD_HOME),
                              hold_time=1.0, ee_site=None)
            finite = bool(np.all(np.isfinite(data.qpos))
                          and np.all(np.isfinite(data.qvel)))
            out["settle"] = dict(
                end_speed=settle["end_speed"], finite=finite,
                speed_ratio=(settle["end_speed"] / SETTLE_SPEED_MAX
                             if finite else float("inf")),
                ok=finite and settle["end_speed"] < SETTLE_SPEED_MAX)
        except Exception as e:
            out["settle"] = dict(ok=False, error=str(e),
                                 speed_ratio=float("inf"))
    else:
        out["settle"] = dict(ok=False, end_speed=float("inf"),
                             speed_ratio=float("inf"),
                             error=f"model has {model.nv} dofs, expected "
                                   f"{gspec.N_JOINTS}")
    from .collision import motion_clearance
    if out["structure"]["ok"]:
        out["motion_clearance"] = motion_clearance(model)
    else:
        out["motion_clearance"] = dict(ok=False, n_checks=3, n_failed=3,
                                       trajectories=[], problems=["invalid structure"])
    out["ok"] = bool(out["density"]["ok"] and out["inertia_ok"]
                     and out["mass_ok"] and out["link_mass_ok"]
                     and out["structure"]["ok"] and out["settle"]["ok"]
                     and out["motion_clearance"]["ok"])
    return out
