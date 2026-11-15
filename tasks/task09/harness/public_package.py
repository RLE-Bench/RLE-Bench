"""The public `harness/` helper package shipped beside each variant's model.

Built into the agent workspace and, at scoring time, next to the sandboxed
trim.py so submissions import the same helpers in both places.
"""
from __future__ import annotations

import json
import os
import shutil

from .variants import VARIANTS

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_FILES = ("__init__.py", "torques.py", "probe_protocol.py")
AGENT_RENAMES = {
    "balance_codesign_public.py": "balance.py",
    "scenarios_public.py": "scenarios.py",
}


def public_spec(name: str) -> str:
    variant = VARIANTS[name]
    with open(os.path.join(HERE, "lead_geometry.json")) as stream:
        geometry = json.load(stream)[name]
    return f'''\
"""Public constants for the {variant.follower_name} GELLO co-design."""
VARIANT_NAME = {variant.name!r}
FOLLOWER_NAME = {variant.follower_name!r}
FOLLOWER_CHAIN = {variant.follower_chain!r}
FOLLOWER_EE = {variant.follower_ee!r}
N_JOINTS = {variant.n_joints}
LEAD_HOME = {variant.home!r}
WORKSPACE_HALFWIDTH = {variant.workspace_halfwidth!r}
SERVO_TAU_NM = 0.35
SERVO_KP = 8.0
SERVO_KD = 0.4
JOINT_DAMPING = 0.01
JOINT_FRICTIONLOSS = 0.03
JOINT_ARMATURE = 0.002
SERVO_MASS_KG = 0.018
LEAD_CHAIN = {geometry['chain']!r}
LEAD_EE = {geometry['ee']!r}
LEAD_BASE_QUAT = {geometry['base_quat']!r}
STOCK_BODY_MASS = {geometry['stock_mass']!r}
MASS_BUDGET_KG = 2.30
TIMESTEP = 0.002
GRAVITY = (0.0, 0.0, -9.81)
JOINT_NAMES = tuple(f"lead_joint{{i}}" for i in range(1, N_JOINTS + 1))
EE_SITE = "lead_ee_site"
'''


def write_package(package: str, name: str) -> None:
    """Write the public package for variant `name` into directory `package`."""
    os.makedirs(package)
    for filename in AGENT_FILES:
        shutil.copy(os.path.join(HERE, filename), os.path.join(package, filename))
    for source_name, destination_name in AGENT_RENAMES.items():
        shutil.copy(os.path.join(HERE, source_name),
                    os.path.join(package, destination_name))
    with open(os.path.join(package, "spec.py"), "w", encoding="utf-8") as stream:
        stream.write(public_spec(name))
