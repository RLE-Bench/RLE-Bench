"""Regression tests for the Antigravity agent adapter."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from harbor.models.agent.context import AgentContext

from rlebench.agents.antigravity import ClosedBookAntigravityCli

_KEY = {"GEMINI_API_KEY": "test-key"}


@pytest.fixture(autouse=True)
def _no_host_credentials(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GEMINI_BASE_URL", "AGY_ADC_AUTH"):
        monkeypatch.delenv(name, raising=False)


def _agent(tmp_path: Path, **kwargs: Any) -> ClosedBookAntigravityCli:
    kwargs.setdefault("extra_env", dict(_KEY))
    return ClosedBookAntigravityCli(
        logs_dir=tmp_path / "logs", model_name="gemini/gemini-3.7-flash", **kwargs
    )


def _record(agent: ClosedBookAntigravityCli) -> list[tuple[str, dict[str, str] | None]]:
    calls: list[tuple[str, dict[str, str] | None]] = []

    async def record_exec(environment: Any, *, command: str, env=None, **kwargs: Any):
        calls.append((command, env))
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    agent.exec_as_agent = record_exec  # type: ignore[method-assign]
    return calls


def _agy_runs(calls) -> list[tuple[str, dict[str, str] | None]]:
    return [(c, env) for c, env in calls if "/agy --dangerously-skip-permissions" in c]


@pytest.mark.parametrize("extra_env, provider", [(_KEY, "gemini"), ({}, None)])
def test_api_key_selects_gemini_provider(tmp_path, extra_env, provider):
    config = _agent(tmp_path, extra_env=extra_env)._build_settings_config("gemini-3.7-flash")
    assert config.get("modelProvider") == provider


def test_run_command_and_env(tmp_path: Path) -> None:
    agent = _agent(tmp_path, reasoning_effort="high")
    calls = _record(agent)
    asyncio.run(agent.run("finish the task", object(), AgentContext()))

    [(command, env)] = _agy_runs(calls)
    assert "--print-timeout 24h" in command and "--continue" not in command
    assert "--model gemini-3.7-flash" in command
    assert "--effort high" in command
    assert "--prompt='finish the task'" in command
    assert env == {"GEMINI_API_KEY": "test-key", "GEMINI_CLI_TRUST_WORKSPACE": "true"}


def test_resume_continues_the_conversation(tmp_path: Path) -> None:
    assert ClosedBookAntigravityCli.capabilities.resume
    agent = _agent(tmp_path)
    calls = _record(agent)
    asyncio.run(agent.run("develop", object(), AgentContext()))
    asyncio.run(agent.resume("evaluate", object(), AgentContext()))

    first, second = (c for c, _ in _agy_runs(calls))
    assert "--continue" not in first
    assert "--continue" in second and "--print-timeout 24h" in second
    assert "evaluate" in second and not agent._resume


@pytest.mark.parametrize("explicit, policy, expected", [
    ({}, True, True), ({}, False, False),
    ({"disable_web_search": False}, True, False), ({"disable_web_search": True}, False, True),
])
def test_network_policy_drives_web_switch(tmp_path, explicit, policy, expected):
    agent = _agent(tmp_path, **explicit)
    asyncio.run(agent.set_web_tools_disabled(policy, object()))
    assert agent.options.disable_web_search is expected


class LocalEnvironment:
    """Exercise Harbor's real execution/error wrapper in an isolated home."""

    _NOOP = ("/agy --dangerously-skip-permissions", "agy models", ".antigravity-run-start",
             "antigravity-cli/brain")

    def __init__(self, home: Path, fail: str | None = None):
        self.home = home
        self.fail = fail
        self.calls: list[str] = []

    async def exec(self, command, *, user=None, env=None, **kwargs):
        self.calls.append(command)
        assert user is None  # Harbor must retain the environment's agent user.
        if self.fail and self.fail in command:
            return SimpleNamespace(return_code=1, stdout="", stderr="injected failure")
        if any(marker in command for marker in self._NOOP):
            return SimpleNamespace(return_code=0, stdout="", stderr="")
        result = subprocess.run(
            ['bash', '-c', command], capture_output=True, text=True,
            env={**os.environ, **(env or {}), 'HOME': str(self.home)},
        )
        return SimpleNamespace(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)


@pytest.mark.parametrize('original', [None, '{"existing":{"enabled":false}}\n', '{broken'])
def test_hook_lifecycle(tmp_path, original):
    agent = _agent(tmp_path, disable_web_search=True)
    path = tmp_path / '.gemini/config/hooks.json'
    path.parent.mkdir(parents=True)
    if original is not None:
        path.write_text(original)
    environment = LocalEnvironment(tmp_path)
    if original == '{broken':
        with pytest.raises(RuntimeError):
            asyncio.run(agent.run('test', environment, AgentContext()))
        assert not any('/agy --dangerously' in c for c in environment.calls)
    else:
        asyncio.run(agent.exec_as_agent(environment, command=agent._build_web_hook_command()))
        config = json.loads(path.read_text())
        hook = config[agent._WEB_HOOK_NAME]['PreToolUse'][0]
        for tool in ('search_web', 'read_url_content'):
            assert re.fullmatch(hook['matcher'], tool)
        for tool in ('run_command', 'view_file'):
            assert not re.fullmatch(hook['matcher'], tool)
        result = subprocess.run(['sh', '-c', hook['hooks'][0]['command']], capture_output=True, text=True, check=True)
        assert json.loads(result.stdout)['decision'] == 'deny'
        asyncio.run(agent.exec_as_agent(environment, command=agent._build_restore_web_hook_command()))
    assert path.read_text() == original if original is not None else not path.exists()
    assert not list(path.parent.glob('hooks.json.rlebench-*'))


@pytest.mark.parametrize('disable', [True, False])
def test_web_control_and_cleanup(tmp_path, disable):
    agent = _agent(tmp_path, disable_web_search=disable)
    environment = LocalEnvironment(tmp_path)
    asyncio.run(agent.run('test', environment, AgentContext()))
    assert any(agent._build_web_hook_command() in c for c in environment.calls) == disable
    assert any('/agy --dangerously' in c for c in environment.calls)
    assert not (tmp_path / '.gemini/config/hooks.json').exists()


@pytest.mark.parametrize('failure', ['d.update', 'settings.json', '/agy --dangerously'])
def test_failure_cleans_hooks(tmp_path, failure):
    agent = _agent(tmp_path, disable_web_search=True)
    path = tmp_path / '.gemini/config/hooks.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"existing":{}}\n')
    environment = LocalEnvironment(tmp_path, fail=failure)
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run('test', environment, AgentContext()))
    assert path.read_text() == '{"existing":{}}\n'
    if failure != '/agy --dangerously':
        assert not any('/agy --dangerously' in c for c in environment.calls)


def test_cancellation_restores_hook(tmp_path):
    agent = _agent(tmp_path, disable_web_search=True)
    environment = LocalEnvironment(tmp_path)
    execute = environment.exec

    async def cancel(command, **kwargs):
        if '/agy --dangerously' in command:
            raise asyncio.CancelledError
        return await execute(command, **kwargs)

    environment.exec = cancel
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(agent.run('test', environment, AgentContext()))
    assert not (tmp_path / '.gemini/config/hooks.json').exists()


def test_cleanup_failure_is_reported(tmp_path):
    agent = _agent(tmp_path, disable_web_search=True)
    environment = LocalEnvironment(tmp_path, fail='shutil.rmtree')
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run('test', environment, AgentContext()))


def test_partial_hook_install_preserves_original(tmp_path):
    agent = _agent(tmp_path, disable_web_search=True)
    path = tmp_path / '.gemini/config/hooks.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"existing":{}}\n')
    command = agent._build_web_hook_command().replace("b.mkdir();", "b.mkdir(); raise RuntimeError();")
    agent._build_web_hook_command = lambda: command
    with pytest.raises(RuntimeError):
        asyncio.run(agent.run('test', LocalEnvironment(tmp_path), AgentContext()))
    assert path.read_text() == '{"existing":{}}\n'
    assert not list(path.parent.glob('hooks.json.rlebench-*'))


def test_cleanup_error_preserves_original_failure(tmp_path):
    agent = _agent(tmp_path, disable_web_search=True)
    environment = LocalEnvironment(tmp_path, fail='shutil.rmtree')
    execute = environment.exec

    async def fail_run(command, **kwargs):
        if '/agy --dangerously' in command:
            raise ValueError('original failure')
        return await execute(command, **kwargs)

    environment.exec = fail_run
    with pytest.raises(ValueError, match='original failure'):
        asyncio.run(agent.run('test', environment, AgentContext()))
