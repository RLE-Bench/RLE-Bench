"""RLE operations-layer contracts for the task05 nanoVLA family."""

from rlebench.repo import families, resolve


SUBTASKS = {"01-libero-open-design", "02-libero-robustness",
            "03-robotwin-open-design", "04-robotwin-robustness"}


def test_task05_is_a_four_target_subtasks_family():
    family = families()["task05"]
    assert family.manifest["task"]["layout"] == "subtasks"
    assert {target.name.rsplit("/", 1)[-1] for target in family.targets()} == SUBTASKS


def test_unique_bare_subtask_name_resolves():
    target = resolve("03-robotwin-open-design")[0]
    assert target.name == "task05/03-robotwin-open-design"
    assert target.path == "tasks/task05/03-robotwin-open-design"
    assert target.instance is None


def test_every_subtask_takes_one_gpu_from_the_default_slot_placement():
    """`rlebench run --device cuda:N` fills ${RLEBENCH_GPU} per cell."""
    family = families()["task05"]
    assert family.run_cfg["gpu"] is True
    for target in family.targets():
        compose = (family.dir / target.name.rsplit("/", 1)[-1] /
                   "environment" / "docker-compose.yaml").read_text()
        assert "${RLEBENCH_GPU:-0}" in compose
        assert "NANOVLA_GPU_" not in compose
