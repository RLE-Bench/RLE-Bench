"""Endpoint configs: credentials, endpoints, egress hosts, agent flags.

`rlebench run -a <scaffold> -m <provider>/<model_name> -e <vendor>/<lane>`:
the scaffold is the CLI Harbor drives; the config selects its endpoint and
credentials. A recipe is one (agent, config) pair. Task-side extras (resume, which
egress hosts apply, the closed-book switch) are not here -- runner.build adds them
from the family manifest and the target's task.toml.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass
class Resolved:
    agent: str
    model: str
    config: str = ""
    harbor_model: str = ""  # endpoint-specific spelling when Harbor needs a bare model
    agent_dir: str = ""  # output subdir label (differs from agent for module-path agents)
    api_hosts: list[str] = field(default_factory=list)  # model API, open during the agent phase
    setup_hosts: list[str] = field(default_factory=list)  # agent install, open under a restricted environment
    agent_envs: list[str] = field(default_factory=list)  # repeated --agent-env values, in order
    agent_kwargs: list[str] = field(default_factory=list)  # repeated --ak values (non-closed-book)
    closed_book: list[str] = field(default_factory=list)  # trailing --ak closed-book switch
    open_book: list[str] = field(default_factory=list)  # trailing --ak values when --open-book
    exports: dict[str, str] = field(default_factory=dict)
    unsets: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    retries_default: int | None = None  # --max-retries when the user passes none

    def __post_init__(self) -> None:
        self.agent_dir = self.agent_dir or self.agent

    def subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        for k in self.unsets:
            env.pop(k, None)
        env.update(self.exports)
        return env


def _need_one(env: dict[str, str], *names: str) -> str:
    for n in names:
        if env.get(n):
            return n
    raise SystemExit(f"error: set one of: {' '.join(names)}")


def _url_host(url: str) -> str:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port  # Validate the port even though the allowlist uses hosts.
        if (parsed.scheme in {"http", "https"} and host
                and not any(c.isspace() for c in url)
                and parsed.username is None and parsed.password is None
                and not parsed.query and not parsed.fragment
                and (port is None or port > 0)):
            return host
    except ValueError:
        pass
    raise SystemExit("error: API base URL must be an http(s) URL with a host and optional port/path, without credentials, query or fragment")


def _codex_base_url(env: dict[str, str]) -> str | None:
    return env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE") or None


_CLAUDE_SETUP_HOSTS = ["deb.debian.org", "downloads.claude.ai"]
_CODEX_SETUP_HOSTS = [
    "deb.debian.org",
    "raw.githubusercontent.com",
    "github.com",
    "objects.githubusercontent.com",
    "nodejs.org",
    "registry.npmjs.org",
]
# Harbor disables native web tools on a restricted network unless told otherwise.
_WEB = {"closed_book": ["disable_web_search=true"], "open_book": ["disable_web_search=false"]}
_ANTIGRAVITY = "rlebench.agents.antigravity"
_GROK = "rlebench.agents.grok"

# --- claude-code ----------------------------------------------------------------


def _cc_subscription(model: str, env: dict[str, str]) -> Resolved:
    """Claude Code subscription (OAuth); inherited API config is dropped."""
    _need_one(env, "CLAUDE_CODE_OAUTH_TOKEN")
    return Resolved(
        "claude-code", model,
        api_hosts=["api.anthropic.com"],
        setup_hosts=_CLAUDE_SETUP_HOSTS + ["api.anthropic.com", "statsig.anthropic.com"],
        **_WEB,
        exports={"CLAUDE_FORCE_OAUTH": "1"},
        unsets=["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"],
        retries_default=2,  # transient 429s kill a whole trial on this lane
    )


def _cc_api(model: str, env: dict[str, str]) -> Resolved:
    _need_one(env, "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    base = env.get("ANTHROPIC_BASE_URL")
    host = _url_host(base or "https://api.anthropic.com")
    return Resolved(
        "claude-code", model, api_hosts=[host],
        setup_hosts=_CLAUDE_SETUP_HOSTS + ["statsig.anthropic.com"],
        **_WEB,
        unsets=[
            "CLAUDE_FORCE_OAUTH", "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_USE_BEDROCK", "AWS_BEARER_TOKEN_BEDROCK",
        ] if base else [],
    )


def _cc_compat(prefix: str, base: str, key_vars: str | tuple[str, ...], cred_var: str,
               context_tokens: str = "1000000"):
    """A third-party Anthropic-compatible endpoint through the Claude Code harness (1M context by default, high effort)."""
    key_vars = (key_vars,) if isinstance(key_vars, str) else key_vars

    def recipe(model: str, env: dict[str, str]) -> Resolved:
        key = _need_one(env, *key_vars)
        model = model.removeprefix(f"{prefix}/")
        settings = {"ANTHROPIC_BASE_URL": base, "CLAUDE_CODE_MAX_CONTEXT_TOKENS": context_tokens}
        return Resolved(
            "claude-code", model, api_hosts=[_url_host(base)],
            setup_hosts=_CLAUDE_SETUP_HOSTS,
            agent_envs=[f"{cred_var}=${{{cred_var}}}", *(f"{k}={v}" for k, v in settings.items())],
            agent_kwargs=["reasoning_effort=high"],
            **_WEB,
            exports={cred_var: env[key], **settings},
            unsets=[
                v for v in (
                    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                    "CLAUDE_FORCE_OAUTH", "CLAUDE_CODE_OAUTH_TOKEN",
                    "CLAUDE_CODE_USE_BEDROCK", "AWS_BEARER_TOKEN_BEDROCK",
                ) if v != cred_var
            ],
        )
    return recipe


def _cc_zai(base: str):
    return _cc_compat("zai", base, ("ZAI_API_KEY", "GLM_API_KEY"), "ANTHROPIC_AUTH_TOKEN")


def _cc_kimi(base: str, key_var: str, cred_var: str):
    return _cc_compat("kimi", base, key_var, cred_var)

# --- codex ----------------------------------------------------------------------


def _codex_api(model: str, env: dict[str, str]) -> Resolved:
    _need_one(env, "OPENAI_API_KEY")
    base = _codex_base_url(env)
    host = _url_host(base or "https://api.openai.com")
    extra = ["auth.openai.com"] if host == "api.openai.com" else []
    return Resolved(
        "codex", model, api_hosts=[host],
        setup_hosts=_CODEX_SETUP_HOSTS + extra,
        **_WEB,
        unsets=["CODEX_FORCE_AUTH_JSON", "CODEX_AUTH_JSON_PATH"],
    )


def _codex_subscription(model: str, env: dict[str, str]) -> Resolved:
    """Codex subscription: `codex login` first, harbor ships ~/.codex/auth.json."""
    _need_one(env, "CODEX_FORCE_AUTH_JSON")
    return Resolved(
        "codex", model, api_hosts=["chatgpt.com"],
        setup_hosts=_CODEX_SETUP_HOSTS + ["chatgpt.com", "auth.openai.com"],
        **_WEB,
        unsets=["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"],
    )

# --- agy ------------------------------------------------------------------------


def _agy_api(model: str, env: dict[str, str]) -> Resolved:
    _need_one(env, "GEMINI_API_KEY")
    # Harbor resolves only the `gemini/` spelling to a bare agy model name.
    return Resolved(
        f"{_ANTIGRAVITY}:ClosedBookAntigravityCli", f"gemini/{model.partition('/')[2]}",
        agent_dir="antigravity-cli",
        api_hosts=["generativelanguage.googleapis.com"],
        setup_hosts=[
            "deb.debian.org", "antigravity.google",
            "antigravity-cli-auto-updater-974169037036.us-central1.run.app",
            "storage.googleapis.com",
        ],
        agent_kwargs=["reasoning_effort=high"],
        **_WEB,
        unsets=[
            "AGY_AUTH_JSON_PATH", "AGY_FORCE_AUTH_JSON", "GOOGLE_API_KEY",
            "GOOGLE_GEMINI_BASE_URL", "GOOGLE_GENAI_API_VERSION",
            "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION",
        ],
    )


# --- grok-build -----------------------------------------------------------------

_GROK_SETUP_HOSTS = ["deb.debian.org", "x.ai", "storage.googleapis.com"]


def _grok_api(model: str, env: dict[str, str]) -> Resolved:
    """xAI's grok CLI; harbor strips the `xai/` prefix."""
    _need_one(env, "XAI_API_KEY")
    return Resolved(
        f"{_GROK}:ResumingGrokBuild", model, agent_dir="grok-build",
        api_hosts=["api.x.ai"], setup_hosts=_GROK_SETUP_HOSTS, **_WEB,
    )


def _grok_subscription(model: str, env: dict[str, str]) -> Resolved:
    """Grok subscription: `grok login` first, the adapter ships ~/.grok/auth.json."""
    _need_one(env, "GROK_FORCE_AUTH_JSON", "GROK_AUTH_JSON_PATH")
    return Resolved(
        f"{_GROK}:SubscriptionGrokBuild", model, agent_dir="grok-build",
        api_hosts=["cli-chat-proxy.grok.com", "auth.x.ai", "accounts.x.ai", "api.x.ai"],
        setup_hosts=_GROK_SETUP_HOSTS, unsets=["XAI_API_KEY"], **_WEB,
    )


# --- the table ------------------------------------------------------------------

AGENT_ALIASES = {
    "cc": "claude-code",
    "codex-cli": "codex",
    "grok": "grok-build",
}

DEFAULT_CONFIG = {
    "claude-code": "anthropic/api",
    "codex": "openai/api",
    "agy": "gemini/api",
    "grok-build": "xai/api",
}

RECIPES = {
    ("claude-code", "anthropic/claude-code"): _cc_subscription,
    ("claude-code", "anthropic/api"): _cc_api,
    ("claude-code", "zai/api"): _cc_zai("https://api.z.ai/api/anthropic"),
    ("claude-code", "zai/api-cn"): _cc_zai("https://open.bigmodel.cn/api/anthropic"),
    ("claude-code", "kimi/kimi-code"): _cc_kimi("https://api.kimi.com/coding/", "KIMI_API_KEY", "ANTHROPIC_API_KEY"),
    ("claude-code", "kimi/api"): _cc_kimi("https://api.moonshot.ai/anthropic", "MOONSHOT_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    ("claude-code", "kimi/api-cn"): _cc_kimi("https://api.moonshot.cn/anthropic", "MOONSHOT_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    ("claude-code", "deepseek/api"): _cc_compat("deepseek", "https://api.deepseek.com/anthropic", "DEEPSEEK_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    ("codex", "openai/codex"): _codex_subscription,
    ("codex", "openai/api"): _codex_api,
    ("agy", "gemini/api"): _agy_api,
    ("grok-build", "xai/api"): _grok_api,
    ("grok-build", "xai/grok-build"): _grok_subscription,
}

CONFIGS = sorted({config for _, config in RECIPES})


def configs_for(agent: str) -> list[str]:
    return [config for a, config in RECIPES if a == agent]


def resolve(agent: str, model: str | None, config: str | None = None,
            env: dict[str, str] | None = None) -> Resolved:
    """(agent, config, model) -> the harbor agent plus its endpoint recipe.

    `oracle` needs neither model nor config. An agent without a builtin config
    passes through to harbor untouched (bring your own hosts via `-- ...`)."""
    env = dict(os.environ) if env is None else env
    agent = AGENT_ALIASES.get(agent, agent)
    if agent == "oracle":
        if config:
            raise SystemExit("error: the oracle takes no -e endpoint")
        return Resolved("oracle", model or "")
    if not model:
        raise SystemExit(f"error: -m/--model is required for {agent}")
    selected_config = config or DEFAULT_CONFIG.get(agent)
    if (agent, selected_config) == ("agy", "gemini/api") and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:-]*", model
    ):
        model = f"google/{model}"
    custom_base = {
        ("codex", "openai/api"): _codex_base_url(env),
        ("claude-code", "anthropic/api"): env.get("ANTHROPIC_BASE_URL"),
    }.get((agent, selected_config))
    bare_model = custom_base and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", model)
    provider, separator, name = model.partition("/")
    if not bare_model and (not separator or not provider or not name or any(c.isspace() for c in model)):
        raise SystemExit("error: -m/--model must be provider/model_name, or a served-model alias with a custom API base URL")
    known = set(DEFAULT_CONFIG)
    if config is None:
        if agent not in known:
            return Resolved(agent, model or "", warnings=[
                f"no builtin config for agent {agent!r}; hosts and credentials are yours (`-- --allow-agent-host ...`)"
            ])
        config = DEFAULT_CONFIG[agent]
    recipe = RECIPES.get((agent, config))
    if recipe is None:
        if agent in known:
            raise SystemExit(f"error: no config {config!r} for {agent}; choices: {', '.join(configs_for(agent))}")
        raise SystemExit(f"error: unknown agent {agent!r} for -e; agents with configs: {', '.join(sorted(known))}")
    resolved = recipe(model, env)
    resolved.harbor_model = resolved.model
    resolved.model = model
    resolved.config = config
    return resolved
