"""The rlebench CLI: target resolution, configs, and the harbor argv per lane.

`rlebench run <target>... -a <agent> [-m model] [-e vendor/lane] [--device dev...]`.
rlebench owns only what harbor cannot know about a task (egress lane, GPU
override, resume, closed-book, dataset mount, device placement); everything
else is passed through. A recipe change is a behavior change: update the
expectations here with one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rlebench import cli, providers  # noqa: E402
from rlebench.repo import families, resolve  # noqa: E402

# Emit only the task01 level and task02 group these CLI tests exercise.
_EMITTERS = (
    ("tasks/task01/L1", "tasks/task01/build_levels.py", ("--emit", "L1")),
    ("tasks/task02/06-setting-the-table", "tasks/task02/build_groups.py",
     ("--emit", "06-setting-the-table")),
)


IN_TREE_ASSETS = str(REPO / "third_party" / "robocasa" / "robocasa" / "models" / "assets")


@pytest.fixture(scope="session", autouse=True)
def _matrices_emitted():
    for probe, emitter, args in _EMITTERS:
        if probe == "tasks/task01/L1" or not (REPO / probe).is_dir():
            # A selected-cell build may leave this level only partly emitted.
            subprocess.run([sys.executable, emitter, *args], cwd=REPO, check=True)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("ROBOCASA_ASSET_DIR", "ROBOCASA_VENDOR", "RLEBENCH_GPU", "OPENAI_BASE_URL",
                "ANTHROPIC_BASE_URL", "CODEX_FORCE_AUTH_JSON"):
        monkeypatch.delenv(var, raising=False)


# --- targets ----------------------------------------------------------------------


def test_family_prefix_and_leaf_resolution():
    every = resolve("task01")  # whatever levels are emitted on this checkout
    assert every == families()["task01"].targets() and every
    l1 = resolve("task01/L1")
    assert len(l1) == 5 and all(t.name.startswith("task01/L1/") for t in l1)
    (leaf,) = resolve("task01/L1/01-open-fridge")
    assert (leaf.path, leaf.instance) == ("tasks/task01/L1", "01-open-fridge")
    (group,) = resolve("task02/06-setting-the-table")
    assert (group.path, group.instance) == ("tasks/task02", "06-setting-the-table")
    (single,) = resolve("task08")
    assert (single.path, single.instance) == ("tasks/task08", None)
    (bare,) = resolve("rgb-only")  # unique child name still works
    assert bare.path == "tasks/task06/rgb-only"
    assert resolve("task01/L1/") == l1


def test_unknown_targets_refuse():
    with pytest.raises(SystemExit):
        resolve("task01/L9")
    with pytest.raises(SystemExit):
        resolve("task99")


# --- configs ----------------------------------------------------------------------


def test_config_defaults_and_aliases():
    r = providers.resolve("cc", "anthropic/claude-opus-5", None, {"ANTHROPIC_API_KEY": "k"})
    assert (r.agent, r.config) == ("claude-code", "anthropic/api")
    r = providers.resolve("codex", "openai/gpt-5.6-sol", None, {"OPENAI_API_KEY": "k"})
    assert (r.agent, r.config, r.api_hosts) == ("codex", "openai/api", ["api.openai.com"])
    r = providers.resolve("agy", "gemini-3.7-flash", None, {"GEMINI_API_KEY": "k"})
    assert r.agent.endswith(":GeminiApiAntigravityCli") and r.config == "gemini/api"
    assert r.model == "google/gemini-3.7-flash"
    r = providers.resolve("oracle", None, None, {})
    assert (r.agent, r.model, r.closed_book) == ("oracle", "", [])


def test_invalid_pairs_and_missing_model_refuse():
    for model in ("gpt-5.6-sol", "openai/", "/gpt-5.6-sol", "openai/ model"):
        with pytest.raises(SystemExit, match="provider/model_name"):
            providers.resolve("codex", model, None, {"OPENAI_API_KEY": "k"})
    with pytest.raises(SystemExit):
        providers.resolve("codex", "openai/test-model", "gemini/api", {"OPENAI_API_KEY": "k"})
    with pytest.raises(SystemExit):
        providers.resolve("claude-code", None, None, {"ANTHROPIC_API_KEY": "k"})
    with pytest.raises(SystemExit):
        providers.resolve("oracle", None, "anthropic/api", {})
    with pytest.raises(SystemExit):  # credential check
        providers.resolve("codex", "openai/test-model", "openai/codex", {"OPENAI_API_KEY": "k"})


def test_subscription_lanes():
    r = providers.resolve("cc", "anthropic/claude-opus-5", "anthropic/claude-code", {"CLAUDE_CODE_OAUTH_TOKEN": "t"})
    assert r.exports == {"CLAUDE_FORCE_OAUTH": "1"} and r.retries_default == 2
    assert "ANTHROPIC_API_KEY" in r.unsets
    r = providers.resolve("codex", "openai/gpt-5.6-sol", "openai/codex", {"CODEX_FORCE_AUTH_JSON": "1"})
    assert r.api_hosts == ["chatgpt.com"] and "OPENAI_API_KEY" in r.unsets


@pytest.mark.parametrize("config, key, host, cred", [
    ("kimi/kimi-code", "KIMI_API_KEY", "api.kimi.com", "ANTHROPIC_API_KEY"),
    ("kimi/api", "MOONSHOT_API_KEY", "api.moonshot.ai", "ANTHROPIC_AUTH_TOKEN"),
    ("kimi/api-cn", "MOONSHOT_API_KEY", "api.moonshot.cn", "ANTHROPIC_AUTH_TOKEN"),
])
def test_kimi_lanes_through_claude_code(config, key, host, cred):
    r = providers.resolve("claude-code", "kimi/k3[1m]", config, {key: "secret"})
    assert r.api_hosts == [host] and r.model == "kimi/k3[1m]" and r.harbor_model == "k3[1m]"
    assert r.exports[cred] == "secret" and r.exports["ANTHROPIC_BASE_URL"].startswith(f"https://{host}/")
    assert r.agent_envs[0] == f"{cred}=${{{cred}}}"
    assert r.exports["ANTHROPIC_DEFAULT_FABLE_MODEL"] == "k3[1m]"
    with pytest.raises(SystemExit):
        providers.resolve("kimi-cli", "kimi/k3", config, {key: "secret"})


def test_deepseek_lane_through_claude_code():
    r = providers.resolve("cc", "deepseek/deepseek-flash", "deepseek/api", {"DEEPSEEK_API_KEY": "d"})
    assert r.api_hosts == ["api.deepseek.com"] and r.harbor_model == "deepseek-flash"
    assert r.exports["ANTHROPIC_AUTH_TOKEN"] == "d"
    assert r.exports["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert r.exports["ANTHROPIC_DEFAULT_FABLE_MODEL"] == "deepseek-flash"
    assert "ANTHROPIC_API_KEY" in r.unsets and r.agent_envs[-1] == "CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000"
    assert r.agent_kwargs == ["reasoning_effort=high"]
    with pytest.raises(SystemExit):
        providers.resolve("cc", "deepseek/deepseek-flash", "deepseek/api", {"ANTHROPIC_API_KEY": "k"})


def test_grok_build_lane():
    r = providers.resolve("grok", "xai/grok-4.6", None, {"XAI_API_KEY": "x"})
    assert r.agent == "rlebench.agents.grok:ResumingGrokBuild" and r.agent_dir == "grok-build"
    assert r.config == "xai/api" and r.harbor_model == "xai/grok-4.6"
    assert r.api_hosts == ["api.x.ai"] and "x.ai" in r.setup_hosts
    with pytest.raises(SystemExit):
        providers.resolve("grok-build", "xai/grok-4.6", None, {})
    r = providers.resolve("grok", "xai/grok-4.6", "xai/grok-build", {"GROK_FORCE_AUTH_JSON": "1"})
    assert r.agent == "rlebench.agents.grok:SubscriptionGrokBuild" and r.agent_dir == "grok-build"
    assert "cli-chat-proxy.grok.com" in r.api_hosts and r.unsets == ["XAI_API_KEY"]
    with pytest.raises(SystemExit):
        providers.resolve("grok", "xai/grok-4.6", "xai/grok-build", {"XAI_API_KEY": "x"})


def test_zai_lanes():
    r = providers.resolve("cc", "zai/glm-5.2", "zai/api", {"ZAI_API_KEY": "z"})
    assert r.api_hosts == ["api.z.ai"] and r.exports["ANTHROPIC_AUTH_TOKEN"] == "z"
    r = providers.resolve("cc", "zai/glm-5.2", "zai/api-cn", {"GLM_API_KEY": "z"})
    assert r.api_hosts == ["open.bigmodel.cn"]
    r = providers.resolve("cc", "zai/glm-5.3-flash", "zai/api", {"ZAI_API_KEY": "z"})
    assert r.model == "zai/glm-5.3-flash" and r.harbor_model == "glm-5.3-flash"
    assert r.exports["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1000000" and r.agent_kwargs == ["reasoning_effort=high"]
    assert "ANTHROPIC_API_KEY" in r.unsets and r.exports["ANTHROPIC_DEFAULT_FABLE_MODEL"] == "glm-5.3-flash"


# --- argv per lane ----------------------------------------------------------------


def _run(argv: list[str], monkeypatch, **env: str):
    """(targets, invocations, workers) for `rlebench run <argv>`."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    ns = cli.parse(["run", *argv])
    return cli.build_run(ns, ns.extra)


def test_environment_host_lane_task08(monkeypatch):
    targets, (inv,), workers = _run(["task08", "-a", "claude-code", "-m", "anthropic/claude-opus-5"], monkeypatch,
                                    ANTHROPIC_API_KEY="k")
    assert workers == 1
    assert inv.argv == [
        "harbor", "run", "-p", "tasks/task08", "-a", "claude-code", "-m", "anthropic/claude-opus-5",
        "-o", "jobs/task08/claude-code", "--job-name", "anthropic_claude-opus-5",
        "--allow-environment-host", "deb.debian.org",
        "--allow-environment-host", "downloads.claude.ai",
        "--allow-environment-host", "statsig.anthropic.com",
        "--allow-environment-host", "api.anthropic.com",
        "--ak", "disallowed_tools=WebSearch,WebFetch",
    ]
    assert inv.console_log is None and inv.label.startswith("[cpu] task08 ->")
    assert inv.env["ROBOCASA_ASSET_DIR"] == IN_TREE_ASSETS and "RLEBENCH_DEBUG" not in inv.env


def test_agent_host_lane_task01_with_gpu_pool(monkeypatch):
    targets, invs, workers = _run(
        ["task01/L1", "-a", "codex", "-m", "openai/gpt-5.6-sol", "--device", "cuda:0", "cuda:1"],
        monkeypatch, OPENAI_API_KEY="k",
    )
    assert len(invs) == 5 and workers == 2
    inv = invs[0]
    assert inv.argv == [
        "harbor", "run", "-p", "tasks/task01/L1", "-i", "01-open-fridge",
        "-a", "codex", "-m", "openai/gpt-5.6-sol",
        "-o", "jobs/task01/L1/01-open-fridge/codex", "--job-name", "openai_gpt-5.6-sol",
        "--allow-agent-host", "api.openai.com",
        "--ak", "web_search=disabled",
        "--override-gpus", "0", "--yes",
    ]
    assert [i.env["RLEBENCH_GPU"] for i in invs[:4]] == ["0", "1", "0", "1"]
    assert inv.env["RLEBENCH_DEBUG"] == "1"
    assert inv.console_log == REPO / "jobs/task01/L1/01-open-fridge/codex/openai_gpt-5.6-sol.console.log"


def test_oracle_has_no_model_config_or_resume(monkeypatch):
    _, (t06,), _ = _run(["task06/method-agnostic", "-a", "oracle", "--device", "cuda:2"], monkeypatch)
    assert t06.argv == [
        "harbor", "run", "-p", "tasks/task06/method-agnostic", "-a", "oracle",
        "-o", "jobs/task06/method-agnostic/oracle", "--job-name", "oracle",
        "--override-gpus", "0", "--yes",
    ]
    assert t06.label.startswith("[cuda:any]")
    _, (t01,), _ = _run(["task01/L1/01-open-fridge", "-a", "oracle"], monkeypatch)
    assert "-m" not in t01.argv and "--resume-trajectory" not in t01.argv
    assert t01.env["RLEBENCH_GPU"] == "0"  # default device when a target wants one


def test_device_refusals(monkeypatch):
    with pytest.raises(SystemExit):
        _run(["task01/L1", "-a", "oracle", "--device", "cpu"], monkeypatch)
    with pytest.raises(SystemExit):
        _run(["task08", "-a", "oracle", "--device", "gpu0"], monkeypatch)


def test_cpu_pool_jobs_and_passthrough(monkeypatch):
    _, invs, workers = _run(
        ["task08", "task09", "-a", "oracle", "--device", "cpu", "--jobs", "2", "--force-build", "--", "--no-rebuild", "-n", "1"],
        monkeypatch,
    )
    assert workers == 2 and len(invs) == 2
    assert invs[0].argv[-4:] == ["--force-build", "--no-rebuild", "-n", "1"]
    assert all(i.console_log is not None for i in invs)


def test_open_book_retries_and_subscription_env(monkeypatch):
    _, (inv,), _ = _run(
        ["task08", "-a", "cc", "-e", "anthropic/claude-code", "-m", "anthropic/claude-opus-5", "--open-book"],
        monkeypatch, CLAUDE_CODE_OAUTH_TOKEN="t", ANTHROPIC_API_KEY="stale", HOME="/nonexistent-e1f0",
    )
    assert "--ak" not in inv.argv
    assert inv.argv[inv.argv.index("--max-retries") + 1:][:3] == ["2", "--retry-include", "ApiRateLimitError"]
    assert inv.env.get("CLAUDE_FORCE_OAUTH") == "1" and "ANTHROPIC_API_KEY" not in inv.env
    _, (inv,), _ = _run(["task08", "-a", "cc", "-m", "anthropic/test-model", "--max-retries", "0"], monkeypatch, ANTHROPIC_API_KEY="k")
    assert inv.argv[inv.argv.index("--max-retries") + 1] == "0"


def test_asset_dir_override_wins(monkeypatch):
    _, (inv,), _ = _run(["task01/L1/01-open-fridge", "-a", "oracle"], monkeypatch,
                        ROBOCASA_ASSET_DIR="/mnt/robocasa-assets/v1.0.1")
    assert inv.env["ROBOCASA_ASSET_DIR"] == "/mnt/robocasa-assets/v1.0.1"


def test_double_dash_only_for_run():
    with pytest.raises(SystemExit):
        cli.main(["list", "--", "--x"])


def test_families_have_a_lane():
    assert {f.run_cfg.get("lane") for f in families().values()} == {"agent-host", "environment-host"}


def test_clean_removes_only_the_selected_job_dirs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    jobs = tmp_path / "jobs"
    cell = jobs / "task08" / "claude-code" / "anthropic_claude-opus-5"
    other_model = jobs / "task08" / "claude-code" / "claude-opus-4"
    other_task = jobs / "task09" / "claude-code" / "anthropic_claude-opus-5"
    for d in (cell, other_model, other_task):
        d.mkdir(parents=True)

    assert cli.main(["clean", "task08", "-a", "claude-code", "-m", "anthropic/claude-opus-5", "-o", str(jobs),
                     "--dry-run"]) == 0
    assert "would remove" in capsys.readouterr().out and cell.exists()

    assert cli.main(["clean", "task08", "-a", "claude-code", "-m", "anthropic/claude-opus-5", "-o", str(jobs)]) == 0
    assert not cell.exists() and other_model.exists() and other_task.exists()

    monkeypatch.setenv("ZAI_API_KEY", "z")
    zai = jobs / "task08" / "claude-code" / "zai_glm-5.3-flash"
    zai.mkdir(parents=True)
    args = ["task08", "-a", "claude-code", "-m", "zai/glm-5.3-flash", "-e", "zai/api", "-o", str(jobs)]
    ns = cli.parse(["run", *args])
    _, (inv,), _ = cli.build_run(ns, [])
    assert inv.argv[inv.argv.index("-m") + 1] == "glm-5.3-flash"
    assert inv.argv[inv.argv.index("--job-name") + 1] == zai.name
    assert cli.main(["clean", *args]) == 0
    assert not zai.exists() and other_model.exists()

    # No -a: the whole target, every agent and model.
    assert cli.main(["clean", "task08", "-o", str(jobs)]) == 0
    assert not (jobs / "task08").exists() and other_task.exists()


def test_device_short_flag_refuses_but_harbor_dataset_passes_through(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        cli.parse(["run", "task08", "-a", "oracle", "-d", "cpu"])
    assert exc.value.code == 2
    _, (inv,), _ = _run(
        ["task08", "-a", "oracle", "--device", "cpu", "--", "-d", "dataset@1.0"],
        monkeypatch,
    )
    assert inv.argv[-2:] == ["-d", "dataset@1.0"]


def test_run_has_no_clean_flag():
    with pytest.raises(SystemExit):
        cli.parse(["run", "task08", "-a", "oracle", "--clean"])


@pytest.mark.parametrize("agent_flag", [
    ["-a", "oracle"], ["--agent", "terminus-2"],
    ["--agent=oracle"], ["-aterminus-2"],
])
@pytest.mark.parametrize("separator", [[], ["--"]])
def test_duplicate_agents_refuse(agent_flag, separator, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.parse(["run", "task08", "-a", "oracle", *separator, *agent_flag])
    assert exc.value.code == 2
    assert "agent configs are duplicated" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["run", "clean"])
@pytest.mark.parametrize("flag", ["-e", "--endpoint"])
def test_endpoint_selector(command, flag):
    args = cli.parse([command, "task08", "-a", "cc", flag, "kimi/api"])
    assert args.endpoint == "kimi/api"


@pytest.mark.parametrize("command", ["run", "clean"])
@pytest.mark.parametrize("flag", ["-c", "--config"])
def test_old_endpoint_flags_refuse(command, flag):
    with pytest.raises(SystemExit) as exc:
        cli.parse([command, "task08", "-a", "oracle", flag, "kimi/api"])
    assert exc.value.code == 2


def test_harbor_config_and_agent_options_pass_through(monkeypatch):
    extra = ["-c", "job.yaml", "--agent-env", "MODE=test", "--ak", "temperature=0.2"]
    _, (inv,), _ = _run(["task08", "-a", "oracle", "--", *extra], monkeypatch)
    assert inv.argv[-len(extra):] == extra


@pytest.mark.parametrize('open_book', [False, True])
def test_agy_default_web_control(monkeypatch, open_book):
    _, invocations, _ = _run(
        ['task08', '-a', 'agy', '-m', 'google/gemini-3.7-flash']
        + (['--open-book'] if open_book else []),
        monkeypatch, GEMINI_API_KEY='test-key',
    )
    argv = invocations[0].argv
    assert argv[argv.index('-a') + 1] == 'rlebench.agents.antigravity:GeminiApiAntigravityCli'
    assert ('disable_web_tools=true' in argv) == (not open_book)


@pytest.mark.parametrize('open_book', [False, True])
def test_grok_build_web_control(monkeypatch, open_book):
    _, (inv,), _ = _run(
        ['task08', '-a', 'grok-build', '-m', 'xai/grok-4.6'] + (['--open-book'] if open_book else []),
        monkeypatch, XAI_API_KEY='x',
    )
    assert inv.argv[inv.argv.index('-a') + 1] == 'rlebench.agents.grok:ResumingGrokBuild'
    assert ('disable_web_search=false' in inv.argv) == open_book
    assert ('disable_web_search=true' in inv.argv) == (not open_book)
    assert '--allow-environment-host' in inv.argv and 'api.x.ai' in inv.argv


@pytest.mark.parametrize('model', ['gemini-3.7-flash', 'google/gemini-3.7-flash'])
@pytest.mark.parametrize('open_book', [False, True])
def test_agy_gemini_api(monkeypatch, model, open_book):
    _, invocations, _ = _run(
        ['task08', '-a', 'agy', '-e', 'gemini/api', '-m', model]
        + (['--open-book'] if open_book else []),
        monkeypatch, GEMINI_API_KEY='test-gemini-secret', AGY_FORCE_AUTH_JSON='1',
        AGY_AUTH_JSON_PATH='/unused/token', GOOGLE_API_KEY='other-key',
        GOOGLE_GEMINI_BASE_URL='https://unused.example', GOOGLE_GENAI_USE_VERTEXAI='true',
        GOOGLE_APPLICATION_CREDENTIALS='/unused/credentials',
    )
    inv = invocations[0]
    assert inv.argv[inv.argv.index('-a') + 1] == 'rlebench.agents.antigravity:GeminiApiAntigravityCli'
    assert inv.argv[inv.argv.index('-m') + 1] == 'google/gemini-3.7-flash'
    assert inv.env['GEMINI_API_KEY'] == 'test-gemini-secret'
    assert 'test-gemini-secret' not in inv.display()
    for key in ['AGY_FORCE_AUTH_JSON', 'AGY_AUTH_JSON_PATH', 'GOOGLE_API_KEY',
                'GOOGLE_GEMINI_BASE_URL', 'GOOGLE_GENAI_USE_VERTEXAI', 'GOOGLE_APPLICATION_CREDENTIALS']:
        assert key not in inv.env
    assert ('disable_web_tools=true' in inv.argv) == (not open_book)
    assert 'reasoning_effort=high' in inv.argv
    for host in ['generativelanguage.googleapis.com', 'antigravity.google',
                 'antigravity-cli-auto-updater-974169037036.us-central1.run.app',
                 'storage.googleapis.com']:
        assert host in inv.argv


def test_agy_api_requires_key():
    with pytest.raises(SystemExit, match='GEMINI_API_KEY'):
        providers.resolve('agy', 'gemini-3.7-flash', 'gemini/api', {})
