"""Trusted Panda model assembly for task10."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from . import spec


def _robot_xml() -> Path:
    explicit = os.environ.get("RLEBENCH_TASK10_ROBOT")
    if explicit:
        return Path(explicit)
    repo = Path(__file__).resolve().parents[3]
    return repo / "assets/robots/franka_emika_panda/panda_nohand.xml"


def build_model(payload_scale: float = 1.0, contact_surface: bool = True) -> mujoco.MjModel:
    source = _robot_xml().resolve()
    root = ET.parse(source).getroot()
    root.find("compiler").set("meshdir", str(source.parent / "assets"))
    option = root.find("option")
    option.set("timestep", str(spec.PHYSICS_DT))
    option.set("integrator", "implicitfast")
    option.set("iterations", "50")
    option.set("ls_iterations", "20")
    attachment = root.find(".//body[@name='attachment']")
    ET.SubElement(attachment, "geom", {
        "name": "probe", "type": "sphere", "size": "0.022",
        "mass": "0.08", "rgba": "0.2 0.7 0.9 1", "friction": "0.8 0.02 0.002", "contype": "2", "conaffinity": "4",
    })
    world = root.find("worldbody")
    ET.SubElement(world, "geom", {
        "name": "floor", "type": "plane", "size": "2 2 0.1",
        "rgba": "0.18 0.18 0.18 1", "pos": "0 0 -0.02",
    })
    if contact_surface:
        ET.SubElement(world, "geom", {
            "name": "surface", "type": "box", "size": "0.20 0.24 0.015",
            "pos": "0.54 0 0.385", "rgba": "0.7 0.5 0.2 1", "contype": "4", "conaffinity": "2",
            "solref": "1.0 1", "solimp": "0.1 0.5 0.1", "friction": "0.9 0.02 0.002",
        })
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_gainprm[:, 0] = 1.0
    model.actuator_biasprm[:, :] = 0.0
    model.actuator_ctrlrange[:, 0] = -spec.TORQUE_LIMIT
    model.actuator_ctrlrange[:, 1] = spec.TORQUE_LIMIT
    model.actuator_forcerange[:, 0] = -spec.TORQUE_LIMIT
    model.actuator_forcerange[:, 1] = spec.TORQUE_LIMIT
    model.body_mass[model.body("link7").id] *= payload_scale
    return model


def save_nominal(path: str) -> None:
    model = build_model(1.0)
    mujoco.mj_saveModel(model, path)
