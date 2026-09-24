from pathlib import Path

import pytest

from harness.runtime.handoff import copy_regular, freeze


def test_snapshot_cannot_follow_a_private_symlink(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    secret = tmp_path/'secret';secret.write_text('private')
    (source/'manual').symlink_to(secret)
    with pytest.raises(ValueError):
        copy_regular(source,tmp_path/'snapshot')
    assert not (tmp_path/'snapshot/manual').exists()


def test_snapshot_is_frozen_once(tmp_path):
    private = tmp_path/'private';private.mkdir()
    workspace = tmp_path/'workspace';workspace.mkdir()
    harness = workspace/'agent_harness';harness.mkdir()
    (harness/'MANUAL.md').write_text('development')
    freeze(private,workspace)
    (harness/'MANUAL.md').write_text('evaluation')
    freeze(private,workspace)
    assert (private/'handoff/MANUAL.md').read_text() == 'development'


def test_task_owned_runtime_copies_agree():
    root = Path(__file__).resolve().parents[2]
    expected = root/'tasks/task01/harness'
    for family in ('task02','task03'):
        for original in (expected/'runtime').glob('*.py'):
            assert original.read_bytes() == (root/'tasks'/family/'harness/runtime'/original.name).read_bytes()
        for name in ('backend.py','compat.py'):
            assert (expected/name).read_bytes() == (root/'tasks'/family/'harness'/name).read_bytes()
    assert not (root/'rlebench/core/runtime').exists()
