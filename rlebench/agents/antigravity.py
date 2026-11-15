"""Antigravity with resumable phases and a hook-based web-tool switch."""

from __future__ import annotations

import json
import shlex
import sys
from uuid import uuid4

from harbor.agents.installed.antigravity_cli import AntigravityCli
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class ClosedBookAntigravityCli(AntigravityCli):
    """Upstream Antigravity CLI plus `--continue` resume and web-tool denial.

    agy has no switch for its web tools, so `disable_web_search` (an explicit
    `--ak`, or Harbor's default for a restricted network) installs a
    PreToolUse hook that denies them for the duration of the run.
    """

    _WEB_HOOK_NAME = "harbor-disable-web"

    # agy 1.1.13+ continues the latest workspace conversation with --continue.
    capabilities = AntigravityCli.capabilities.model_copy(
        update={"resume": True, "native_web_tools": True}
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hook_backup = f"hooks.json.rlebench-{uuid4().hex}"

    def build_cli_flags(self) -> str:
        flags = super().build_cli_flags()
        return f"--continue {flags}".strip() if self._resume else flags

    def _build_web_hook_command(self) -> str:
        """Install a temporary hook that blocks only Antigravity's web tools."""
        denial = shlex.quote(
            json.dumps(
                {
                    "decision": "deny",
                    "reason": "Web tools are disabled for this Harbor run.",
                }
            )
        )
        hook = {
            self._WEB_HOOK_NAME: {
                "PreToolUse": [
                    {
                        "matcher": "^(search_web|read_url_content)$",
                        "hooks": [
                            {
                                "type": "command",
                                "command": f"printf %s {denial}",
                            }
                        ],
                    }
                ]
            }
        }
        merge_script = (
            "import json, pathlib, shutil, sys; "
            "p = pathlib.Path.home() / '.gemini/config/hooks.json'; "
            "p.parent.mkdir(parents=True, exist_ok=True); "
            "d = json.loads(p.read_text()) if p.exists() else {}; "
            "d.update(json.loads(sys.argv[1])); "
            f"b = p.with_name({self._hook_backup!r}); b.mkdir(); "
            "shutil.copy2(p, b / 'copy') if p.exists() else (b / 'absent').touch(); "
            "(b / 'copy').replace(b / 'original') if p.exists() else None; "
            "(b / 'new').write_text(json.dumps(d, indent=2) + '\\n'); "
            "(b / 'new').replace(p)"
        )
        return f"python3 -c {shlex.quote(merge_script)} {shlex.quote(json.dumps(hook))}"

    def _build_restore_web_hook_command(self) -> str:
        script = (
            "import pathlib, shutil; "
            "p = pathlib.Path.home() / '.gemini/config/hooks.json'; "
            f"b = p.with_name({self._hook_backup!r}); "
            "(b / 'original').replace(p) if (b / 'original').exists() else "
            "p.unlink(missing_ok=True) if (b / 'absent').exists() else None; "
            "shutil.rmtree(b) if b.exists() else None"
        )
        return f"python3 -c {shlex.quote(script)}"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        hook_attempted = False
        try:
            if self.options.disable_web_search:
                hook_attempted = True
                await self.exec_as_agent(environment, command=self._build_web_hook_command())
            await super().run(instruction, environment, context)
        finally:
            failed = sys.exc_info()[0] is not None
            if hook_attempted:
                try:
                    await self.exec_as_agent(
                        environment, command=self._build_restore_web_hook_command()
                    )
                except Exception as exc:
                    self.logger.warning("Antigravity cleanup failed: %s", exc)
                    if not failed:
                        raise
