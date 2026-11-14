"""Static contract and isolation checks for the task05 nanoVLA subtasks."""

from __future__ import annotations

import ast
import hashlib
import json
import re

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 used by the repository's dev venv.
    import tomli as tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks" / "task05"
SIM = ROOT / "sim" / "libero"
HARNESS = TASKS / "harness"
VERIFIER_BASE = "rlebench-task05-libero-verifier-base:dev"
ROBOTWIN_BASE = "rlebench-task05-robotwin-verifier-base:dev"

# subtask -> (gpus, verifier timeout, encoder bundle, replay budget, plus, agent timeout)
SUBTASKS = {
    "01-libero-open-design": (1, 6000.0, "dinov2", "1800.0", False, 14400.0),
    "02-libero-robustness": (1, 7200.0, "open", "1800.0", True, 14400.0),
    "03-robotwin-open-design": (1, 23400.0, "dinov2", "1800.0", False, 14400.0),
    "04-robotwin-robustness": (1, 23400.0, "open", "1800.0", False, 14400.0),
}
ROBOTWIN = {"03-robotwin-open-design", "04-robotwin-robustness"}
BUNDLES = ("base", "dinov2", "open")


def verifier_base(subtask: str) -> str:
    return ROBOTWIN_BASE if subtask in ROBOTWIN else VERIFIER_BASE


def task_dirs() -> set[str]:
    return {
        path.name
        for path in TASKS.iterdir()
        if path.is_dir() and (path / "task.toml").is_file()
    }


def manifest(subtask: str) -> dict:
    with (TASKS / subtask / "task.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_family_manifest_uses_rlebench_subtasks_layout():
    with (TASKS / "manifest.toml").open("rb") as stream:
        family = tomllib.load(stream)
    assert family["task"]["layout"] == "subtasks"
    assert family["run"] == {"gpu": True}
    assert family["prepare"]["commands"] == [
        ["make", "sim-libero"], ["make", "sim-robotwin"], ["make", "task05"], ["make", "task05-data"],
    ]
    assert set(family["verify"]["images"]) == {VERIFIER_BASE, ROBOTWIN_BASE, "rlebench-robotwin-sim:dev"} | {
        f"rlebench-task05-{kind}-{bundle}:dev" for kind in ("hf", "agent") for bundle in BUNDLES
    } | {f"rlebench-task05-{subtask}-verifier:dev" for subtask in SUBTASKS}


def test_all_four_subtasks_have_complete_harbor_layout():
    assert task_dirs() == set(SUBTASKS)
    required = {
        "task.toml",
        "README.md",
        "instruction.md",
        "environment/docker-compose.yaml",
        "solution/solve.sh",
        "solution/solution.py",
        "tests/Dockerfile",
        "tests/docker-compose.yaml",
        "tests/score_task.py",
        "tests/test.sh",
    }
    for subtask in SUBTASKS:
        present = {
            path.relative_to(TASKS / subtask).as_posix()
            for path in (TASKS / subtask).rglob("*")
            if path.is_file()
        }
        assert required <= present, subtask
    assert (TASKS / "02-libero-robustness" / "tests" / "plus10_manifest.json").is_file()
    for subtask in ROBOTWIN:
        assert (TASKS / subtask / "tests" / "robotwin_test_episodes.json").is_file()


@pytest.mark.parametrize("subtask", sorted(SUBTASKS))
def test_manifest_declares_phase_and_single_file_boundaries(subtask: str):
    gpus, verifier_timeout, bundle, _, _, agent_timeout = SUBTASKS[subtask]
    data = manifest(subtask)
    assert data["schema_version"] == "1.4"
    assert data["task"]["name"] == f"rlebench/task05-nanovla-{subtask}"
    assert data["task"]["version"] == "1.0.0"
    assert data["environment"]["docker_image"] == f"rlebench-task05-agent-{bundle}:dev"
    assert data["environment"]["gpus"] == gpus
    assert data["environment"]["network_mode"] == "public"  # agent setup only; the run phase is the allowlist below
    assert data["metadata"]["difficulty"] == "hard"
    assert data["metadata"]["tags"]
    assert data["agent"] == {"timeout_sec": agent_timeout, "network_mode": "allowlist", "allowed_hosts": []}
    assert data["verifier"]["timeout_sec"] == verifier_timeout
    assert data["verifier"]["network_mode"] == "no-network"
    assert data["verifier"]["environment_mode"] == "separate"
    assert data["verifier"]["environment"]["gpus"] == gpus
    assert data["verifier"]["environment"]["network_mode"] == "no-network"
    assert data["verifier"]["environment"]["docker_image"] == (
        f"rlebench-task05-{subtask}-verifier:dev"
    )
    assert data["verifier"]["collect"] == [
        {
            "service": "main",
            "user": 0,
            "command": "/usr/local/sbin/nanovla-finalize",
            "timeout_sec": 30.0,
        }
    ]
    assert data["artifacts"] == [
        {
            "source": "/logs/artifacts",
            "destination": "_empty-artifacts",
            "exclude": ["*"],
        },
        {
            "source": "/logs/nanovla-handoff/submission/solution.py",
            "destination": "submission/solution.py",
        },
    ]


@pytest.mark.parametrize("subtask", sorted(SUBTASKS))
def test_compose_mounts_exact_assets_and_isolates_the_evaluator(subtask: str):
    gpus, _, _, _, plus, _ = SUBTASKS[subtask]
    shards = "NANOVLA_ROBOTWIN_SHARDS" if subtask in ROBOTWIN else "NANOVLA_SHARDS_L10"
    for phase in ("environment", "tests"):
        text = (TASKS / subtask / phase / "docker-compose.yaml").read_text()
        gpu_ids = set(re.findall(r"RLEBENCH_GPU:-(\d+)", text))
        assert gpu_ids == {str(index) for index in range(gpus)}
        assets = set(re.findall(r"source: \$\{(NANOVLA_[A-Z0-9_]+):\?", text))
        # the LIBERO-plus assets reach the verifier only; development never sees perturbations
        assert assets == {shards} | ({"NANOVLA_LIBERO_PLUS_ASSETS"} if plus and phase == "tests" else set())
        assert "target: /assets/nanovla/shards" in text
        assert "/assets/hf" not in text  # the encoder bundle is baked into the images
        assert not any(path in text for path in ("/scratch", "/nethome", "/dev/shm/")), phase
    agent = (TASKS / subtask / "environment" / "docker-compose.yaml").read_text()
    verifier = (TASKS / subtask / "tests" / "docker-compose.yaml").read_text()
    # the agent container: docker's default caps, the evaluation service beside it
    assert "cap_drop" not in agent
    assert f"image: {verifier_base(subtask)}" in agent
    assert f'"--subtask", "{subtask}"' in agent
    assert "network_mode: none" in agent
    assert agent.count("target: /run/nanovla-eval") == 2
    assert "condition: service_healthy" in agent
    assert 'test: ["CMD", "test", "-S", "/run/nanovla-eval/service.sock"]' in agent
    assert "/opt/LIBERO-plus" not in agent
    # the verifier: hardened, offline, single-file handoff
    assert "cap_drop: [ALL]" in verifier
    assert "read_only: true" in verifier
    assert "/tmp:rw,exec," in verifier
    assert "cap_add: [CHOWN, KILL, SETGID, SETUID]" in verifier
    assert "/logs/artifacts:rw,size=2m,mode=0755" in verifier
    assert "/logs/nanovla-handoff:rw,size=2m,mode=0700" in verifier
    assert "nanovla-eval" not in verifier
    # harbor's docker cp reaches the reward only on a volume
    assert "- type: volume\n        target: /logs/verifier\n" in verifier
    assert "/logs/verifier:" not in verifier
    # harbor leaves /logs/verifier world-writable; root takes it back before scoring
    test_sh = (TASKS / subtask / "tests" / "test.sh").read_text()
    assert test_sh.index("chmod 0755 /logs/verifier") < test_sh.index("score_task.py")


@pytest.mark.parametrize("subtask", sorted(SUBTASKS))
def test_agent_instruction_is_explicit_and_python_is_parseable(subtask: str):
    _, _, bundle, budget, plus, agent_timeout = SUBTASKS[subtask]
    task = TASKS / subtask
    instruction = (task / "instruction.md").read_text()
    assert f"{int(agent_timeout):,}" in instruction
    assert "/logs/artifacts/submission/solution.py" in instruction
    assert "No network" in instruction and "no network during exploration" in instruction
    assert "offline" in instruction.lower()
    assert "reward" in instruction.lower()
    for token in ("NANOVLA_MODE", "NANOVLA_OUTPUT_DIR", "NANOVLA_SOCKET", "ckpt.pt",
                  "weights_only=True", "/run/nanovla-eval/", "nanovla_eval.py", "20,000"):
        assert token in instruction, token
    assert "--set plus" not in instruction
    if subtask in ROBOTWIN:
        assert "solvable" in instruction and "(K, 14)" in instruction and "(16,)" in instruction
    else:
        assert "seeded resets" in instruction
    prose = " ".join(instruction.split())
    if plus:
        assert "the language instruction, the lighting, the object layout, etc." in prose
        assert "perturbed_success_rate" in prose and "LIBERO-plus" not in prose
    if subtask == "04-robotwin-robustness":
        assert "the language instruction, the lighting, the background, etc." in prose
        assert "225 perturbed" in prose and "perturbed_success_rate" in prose
    for relative in ("solution/solution.py", "tests/score_task.py"):
        ast.parse((task / relative).read_text(), filename=f"{subtask}/{relative}")
    scorer = (task / "tests" / "score_task.py").read_text()
    assert f"run_training({budget})" in scorer
    assert "serve_and_evaluate" in scorer
    assert ("--plus-manifest" in scorer) == plus
    if subtask == "01-libero-open-design":
        assert '"--init", "files"' in scorer  # the official initial states
    if subtask in ROBOTWIN:
        assert 'simulator="robotwin"' in scorer and "robotwin_test_episodes.json" in scorer
    dockerfile = (task / "tests" / "Dockerfile").read_text()
    assert dockerfile.startswith(f"FROM {verifier_base(subtask)}")
    assert f"COPY --from=rlebench-task05-hf-{bundle}:dev /assets/hf /assets/hf" in dockerfile
    assert "COPY . /tests/" in dockerfile
    assert "chmod -R go-rwx /tests" in dockerfile


def test_agent_image_carries_no_simulator_and_verifier_is_root_only():
    dockerfile = (SIM / "Dockerfile").read_text()
    agent, verifier = dockerfile.split("# ------------------------------------------------------------- verifier")
    for forbidden in ("LIBERO", "robosuite", "mujoco", "harness/verifier", "libegl"):
        assert forbidden not in agent, forbidden
    assert "COPY sim/libero/requirements-agent.txt" in agent
    assert "COPY tasks/task05/harness/runtime/ /workspace/nanovla/" in agent
    assert "COPY --from=hf /assets/hf /assets/hf" in agent
    assert (SIM / "Dockerfile.hf").read_text().startswith("# One offline Hugging Face bundle")
    assert "USER 1000:1000" in agent
    assert "chmod 0700 /logs/nanovla-handoff" in agent
    for line in (
        "COPY tasks/task05/harness/verifier/ /opt/nanovla-verifier/",
        "chmod -R go-rwx /opt/nanovla-verifier /opt/LIBERO /opt/LIBERO-plus",
        "chown root:65534 /assets/nanovla && chmod 0750 /assets/nanovla",
        "/opt/nanovla-config/libero/config.yaml",
        "/opt/nanovla-config/libero_plus/config.yaml",
    ):
        assert line in verifier, line
    runtime = sorted(path.name for path in (HARNESS / "runtime").iterdir() if path.suffix == ".py")
    assert runtime == ["nanovla_eval.py", "remote_protocol.py", "towers.py"]
    requirements = (SIM / "requirements-agent.txt").read_text()
    for forbidden in ("mujoco", "robosuite", "bddl", "gym"):
        assert forbidden not in requirements


def test_verifier_isolates_training_from_serving():
    common = (HARNESS / "verifier" / "common.py").read_text()
    assert "TRAIN_UID = 65534" in common and "SERVE_UID = 65533" in common
    assert common.index("def run_training") < common.index("def freeze_output") < common.index("def serve_and_evaluate")
    assert "_scratch_sweep()" in common.split("def _prepare_runtime")[1].split("def ")[0]
    assert "st_gid != TRAIN_UID" in common
    assert "O_NOFOLLOW" in common and "follow_symlinks=False" in common
    assert "weights_only=True" in common and "weights_only=False" not in common
    assert 'OUTPUT_FILES = ("ckpt.pt", "meta.json")' in common
    assert "rows.get(key, False)" in common
    finalizer = (HARNESS / "finalize_submission.py").read_text()
    assert "def process_cgroup" in finalizer
    assert 'entries != ["solution.py"]' in finalizer
    assert "signal.SIGKILL" in finalizer
    service = (HARNESS / "verifier" / "eval_service.py").read_text()
    assert "S_ISSOCK" in service and "st_uid != AGENT_UID" in service
    assert "budget" in service
    assert '"--init", "seeded"' in service and "DEV_SEED_BASE = 100_000" in service
    assert "plus" not in service.lower()
    rollout = (HARNESS / "verifier" / "rollout.py").read_text()
    assert "env.seed(args.seed_base + task_id * 1000 + episode)" in rollout
    assert "torch.load(os.path.join(get_libero_path" in rollout
    assert "local_files_only" in (HARNESS / "runtime" / "towers.py").read_text()
    assert "HF model revision mismatch" in (HARNESS / "validate_assets.py").read_text()


def test_tower_tags_keep_their_suffixes():
    """Submissions load tags such as `dinov2_p4` and `siglip2_rp384`."""
    tree = ast.parse((HARNESS / "runtime" / "towers.py").read_text())
    parser = [node for node in tree.body if getattr(node, "name", None) == "_split_tag"
              or any(getattr(target, "id", None) == "_SUFFIX" for target in getattr(node, "targets", []))]
    scope = {"re": re}
    exec(compile(ast.Module(parser, type_ignores=[]), "towers.py", "exec"), scope)
    tags = ("dinov2", "siglip_p4", "siglip2_rp384", "dinov2_p4_rp128")
    assert [scope["_split_tag"](tag) for tag in tags] == [
        ("dinov2", 0), ("siglip", 0), ("siglip2", 384), ("dinov2", 128)]


def test_build_context_and_suite_entrypoints_are_wired():
    dockerignore = (ROOT / ".dockerignore").read_text()
    assert "!sim/libero/**" in dockerignore
    # only harness/ is admitted: solution/ payloads never reach the daemon
    assert "!tasks/task05/harness/**" in dockerignore
    script = (SIM / "libero.sh").read_text()
    for command in ("cmd_base", "cmd_verifiers", "cmd_images"):
        assert command in script, command
    assert "LIBERO_SUBTASKS=(01-libero-open-design 02-libero-robustness 03-robotwin-open-design 04-robotwin-robustness)" in script
    assert "LIBERO_BUNDLES=(base dinov2 open)" in script and "cmd_bundles" in script
    makefile = (ROOT / "Makefile").read_text()
    suites = (ROOT / "tests" / "suites.mk").read_text()
    assert "sim/libero/libero.sh base" in makefile.split("\nsim-libero:\n", 1)[1]
    assert "sim/libero/libero.sh verifiers" in makefile.split("\ntask05: task05-assets\n", 1)[1]
    assert "sim/robotwin/robotwin.sh images" in makefile.split("\nsim-robotwin:\n", 1)[1]
    assert (
        "task05"
        in re.search(r"^TASK_IDS := (.+)$", suites, re.MULTILINE).group(1).split()
    )
    assert "TASK_TESTS_task05" in suites


def test_plus_manifest_is_exact():
    verifier = (TASKS / "subtasks" / "02-libero-robustness" / "tests" / "plus10_manifest.json").read_bytes()
    assert hashlib.sha256(verifier).hexdigest() == "8ec365757825e3f819611bb82fdbb90d61df8967dcb75fd81790c2d124124503"
    entries = json.loads(verifier)
    assert len(entries) == 500 and len({e["bddl"] for e in entries}) == 500
    assert {e["suite"] for e in entries} == {"libero_10"}
    assert sorted({e["task"] for e in entries}) == list(range(10))
    assert all(sum(e["task"] == task for e in entries) == 50 for task in range(10))
    assert all(not e["bddl"].startswith("/") and not e["init"].startswith("/") for e in entries)
    assert not (HARNESS / "verifier" / "plus10_dev_manifest.json").exists()


def test_robotwin_layer_is_root_only_and_agent_free():
    dockerfile = (ROOT / "sim" / "robotwin" / "Dockerfile").read_text()
    for line in (
        "COPY third_party/robotwin/curobo /opt/curobo",
        "COPY third_party/robotwin/assets /opt/robotwin/assets",
        "COPY tasks/task05/harness/verifier/ /opt/nanovla-verifier/",
        "chown root:65534 /assets/nanovla && chmod 0750 /assets/nanovla",
        "chmod -R go-rwx /opt/nanovla-verifier /opt/robotwin",
        "NANOVLA_SIMULATOR=robotwin",
    ):
        assert line in dockerfile, line
    assert "agent" not in dockerfile.split("FROM sim AS verifier")[1].lower().replace("agent-free", "")
    rollout = (HARNESS / "verifier" / "rollout_robotwin.py").read_text()
    assert "expert_solves" in rollout and "--episodes-file" in rollout
    assert 'action_type="qpos"' in rollout
    assert 'CONFIG_FILES = {"clean": "clean50", "randomized": "rand50"}' in rollout
    # the wire protocol carries the simulator's action width: 7 for LIBERO, 14 for RoboTwin
    protocol = (HARNESS / "runtime" / "remote_protocol.py").read_text()
    verifier_common = (HARNESS / "verifier" / "common.py").read_text()
    assert 'ACTION_DIM = int(os.environ.get("NANOVLA_ACTION_DIM", "7"))' in protocol
    assert 'ACTION_DIM = {"robotwin": "14"}.get(SIMULATOR, "7")' in verifier_common
    assert '"NANOVLA_ACTION_DIM": ACTION_DIM' in verifier_common
    for subtask in ROBOTWIN:
        compose = (TASKS / subtask / "environment" / "docker-compose.yaml").read_text()
        assert compose.count('NANOVLA_ACTION_DIM: "14"') == 2
    common = (HARNESS / "verifier" / "common.py").read_text()
    assert "def robotwin_expected" in common and 'script = "rollout_robotwin.py"' in common
    service = (HARNESS / "verifier" / "eval_service.py").read_text()
    assert "robotwin_dev_episodes.json" in service
    for subtask in ROBOTWIN:
        book = json.loads((TASKS / "subtasks" / subtask / "tests" / "robotwin_test_episodes.json").read_text())
        assert set(book) == {"tasks", "episodes"}


# --- RoboTwin: the success check must not depend on state only the expert sets ---

ROBOTWIN_ENVS = Path("third_party/robotwin/envs")


def _self_attrs(node, store: bool) -> set[str]:
    ctx = ast.Store if store else ast.Load
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "self" and isinstance(n.ctx, ctx)}


@pytest.mark.skipif(not ROBOTWIN_ENVS.is_dir(), reason="RoboTwin sources are not vendored")
def test_robotwin_success_checks_need_no_expert_state():
    """check_success may only read attributes the scene setup defines.

    Policy rollouts never call play_once, so an attribute the expert sets there is
    missing at evaluation time and every episode of that task raises. Tasks that
    genuinely need one are restored by rollout_robotwin.expert_state.
    """
    root = Path(__file__).resolve().parent.parent
    names = json.loads((root / "tasks/task05/harness/verifier/robotwin_dev_episodes.json").read_text())["tasks"]
    restored = (root / "tasks/task05/harness/verifier/rollout_robotwin.py").read_text()
    offenders = {}
    for task in names:
        tree = ast.parse((ROBOTWIN_ENVS / f"{task}.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
        if "check_success" not in methods:
            continue
        defined = set()
        for name in ("setup_demo", "load_actors", "__init__", "_init_task_env_", "setup"):
            if name in methods:
                defined |= _self_attrs(methods[name], store=True)
        expert = _self_attrs(methods["play_once"], store=True) if "play_once" in methods else set()
        missing = (_self_attrs(methods["check_success"], store=False) - defined) & expert
        if missing and f'task == "{task}"' not in restored:
            offenders[task] = sorted(missing)
    assert not offenders, f"expert_state must restore: {offenders}"


def test_the_rollout_never_resumes_from_rows_the_agent_could_have_planted():
    """Rows already under /logs/verifier were not written by this verifier."""
    source = (HARNESS / "verifier" / "common.py").read_text()
    body = source[source.index("def run_rollout("):]
    body = body[:body.index("\ndef ")]
    # the pair is cleared before --resume names it, and only RESUME_DIR may seed it
    assert body.index("unlink(missing_ok=True)") < body.index("RESUME_DIR is not None")
    assert body.index("RESUME_DIR is not None") < body.index('"--resume"')
    assert "shutil.rmtree(stale)" in body
