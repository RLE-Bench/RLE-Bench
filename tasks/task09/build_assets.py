"""Stage the three task09 GELLO payloads for one Harbor task."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from rlebench import taskgen  # noqa: E402

from harness.public_package import write_package  # noqa: E402

HARNESS_SRC = os.path.join(REPO, "tasks", "task09", "harness")
CODESIGN_MODELS = os.path.join(HARNESS_SRC, "assets", "gello_codesign")
VARIANT_NAMES = ("franka", "ur5e", "xarm7")

VERIFIER_EXCLUDE = ("assets",)

FORBIDDEN_IN_AGENT = (
    "oracle", "golden", "HIDDEN_SEEDS", "lead_sref", "checkpoint",
    "codesign_battery", "local_provider", "qmat_hold", "qmat_path",
    "qmat_grid", "poke_test", "fixed_hold_configs", "hold_configs",
    "teleop_path", "CodesignEnvelope", "device_buildable",
    "check_validity", "density_report", "link_mass_tol", "payload_max",
    "probe_noise_std", "DENSITY_PRINTED", "DENSITY_ANY",
)


def _assert_public_boundary(root: str) -> None:
    for directory, _dirs, files in os.walk(root):
        for filename in files:
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, root)
            text = open(path, "rb").read().decode("utf-8", errors="replace")
            for token in FORBIDDEN_IN_AGENT:
                if token in relative or token in text:
                    raise SystemExit(
                        f"INVARIANT VIOLATION: {token!r} found in {relative}")


def build_environment_assets() -> None:
    root = os.path.join(HERE, "environment", "assets")
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)
    for name in VARIANT_NAMES:
        source = os.path.join(CODESIGN_MODELS, name)
        workdir = os.path.join(root, name)
        os.makedirs(workdir)
        shutil.copytree(os.path.join(source, "meshes"), os.path.join(workdir, "meshes"))
        shutil.copy(os.path.join(source, "lead_bare.xml"),
                    os.path.join(workdir, "lead.xml"))
        scene = open(os.path.join(source, "scene.xml"), encoding="utf-8").read()
        scene = scene.replace("lead_bare.xml", "lead.xml")
        with open(os.path.join(workdir, "scene.xml"), "w", encoding="utf-8") as stream:
            stream.write(scene)

        write_package(os.path.join(workdir, "harness"), name)
        _assert_public_boundary(workdir)
        print(f"{name} workspace built ({_du(workdir)} KB)")


def build_verifier_context() -> None:
    root = os.path.join(HERE, "tests")
    package = os.path.join(root, "harness")
    if os.path.exists(package):
        shutil.rmtree(package)
    taskgen.clean_copytree(HARNESS_SRC, package, VERIFIER_EXCLUDE)

    models = os.path.join(root, "models", "gello_codesign")
    if os.path.exists(models):
        shutil.rmtree(models)
    for name in VARIANT_NAMES:
        target = os.path.join(models, name)
        os.makedirs(target, exist_ok=True)
        shutil.copy(os.path.join(CODESIGN_MODELS, name, "lead_sref.xml"), target)
        shutil.copytree(os.path.join(CODESIGN_MODELS, name, "meshes"), os.path.join(target, "meshes"))
    print(f"verifier source built ({_du(root)} KB)")


def build_solution_payload() -> None:
    from harness.codesign_oracle import build_reference_submission

    root = os.path.join(HERE, "solution", "payload")
    if os.path.exists(root):
        shutil.rmtree(root)
    for name in VARIANT_NAMES:
        build_reference_submission(os.path.join(root, name), variant=name)
        print(f"{name} Oracle payload built")


def _du(path: str) -> int:
    total = 0
    for directory, _dirs, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(directory, name)) for name in files)
    return total // 1024


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "dev.generate_variant_assets"],
        cwd=REPO, check=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1",
             "PYTHONPATH": os.path.join(REPO, "tasks", "task09")})
    build_environment_assets()
    build_verifier_context()
    build_solution_payload()


if __name__ == "__main__":
    main()
