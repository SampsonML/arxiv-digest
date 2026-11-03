"""Asking Claude questions through the Claude Code CLI (`claude -p`).

arxiv-digest only works with a Claude account (a Claude subscription signed in
to Claude Code), never with API keys or pay-as-you-go API credits:

  - API-key and third-party-provider variables (ANTHROPIC_API_KEY,
    ANTHROPIC_AUTH_TOKEN, CLAUDE_CODE_USE_BEDROCK/VERTEX/FOUNDRY) are removed
    from the environment Claude Code runs in. Claude Code would otherwise
    prefer an API key in the environment over the account sign-in, and bill
    its credits (the cause of "Credit balance is too low" errors);
  - Claude Code counts as ready only when `claude auth status` reports a
    Claude account sign-in (authMethod "claude.ai"), not an Anthropic Console
    (API billing) sign-in.

Each call is also isolated from the person's usual Claude Code setup:

  --tools ""                 no tools: Claude only reads the prompt and answers
  --system-prompt            our instructions replace Claude Code's coding prompt
  --setting-sources ""       their settings and hooks are not loaded
  --no-session-persistence   nothing appears in their session history
  --json-schema              the answer must be JSON matching a schema
  cwd = an empty temp folder no project CLAUDE.md is picked up

(`--bare` would isolate even more, but it only accepts an API key, never a
subscription sign-in.)
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DEFAULT_TIMEOUT = 600  # seconds


class ClaudeError(Exception):
    """Claude Code could not be run or did not give a usable answer."""


class ClaudeNotSignedIn(ClaudeError):
    """Claude Code isn't signed in to a Claude account."""


SIGN_IN_HINT = (
    "Sign in with your Claude account: `claude auth login` in a terminal, or the"
    " Log in button in `arxiv-digest app`."
)
CLAUDE_ACCOUNT = "claude.ai"  # authMethod of a Claude account (subscription) sign-in
# Variables that make Claude Code use an API key or another provider's billing.
API_BILLING_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)
_NOT_SIGNED_IN = re.compile(
    r"not logged in|run /login|invalid api key|oauth token has expired|authentication_error"
    r"|credit balance",
    re.I,
)


def account_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The environment for Claude Code, without anything that selects API billing."""
    env = dict(os.environ if environ is None else environ)
    for name in API_BILLING_VARS:
        env.pop(name, None)
    return env


@dataclass(frozen=True)
class ClaudeStatus:
    """Whether Claude Code is installed and signed in (from `claude auth status`)."""

    installed: bool
    logged_in: bool | None = None  # None: couldn't tell, e.g. an older Claude Code
    email: str | None = None
    plan: str | None = None  # e.g. "pro", "max"
    method: str | None = None  # e.g. "claude.ai"; "console" means API billing

    @property
    def ready(self) -> bool | None:
        """Signed in with a Claude account (None: couldn't tell)."""
        if self.logged_in is None:
            return None
        return bool(self.logged_in and self.method == CLAUDE_ACCOUNT)

    def describe(self) -> str:
        if not self.installed:
            return "Claude Code is not installed"
        if self.logged_in is None:
            return "Claude Code is installed (sign-in status unknown)"
        if not self.logged_in:
            return "Claude Code is installed but not signed in to a Claude account"
        if self.method != CLAUDE_ACCOUNT:
            return (
                "Claude Code is signed in with an Anthropic Console (API billing) account,"
                " which arxiv-digest doesn't use; log in with your Claude account"
            )
        who = f" as {self.email}" if self.email else ""
        plan = f" ({self.plan.capitalize()} plan)" if self.plan else ""
        return f"Signed in to Claude{who}{plan}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "logged_in": self.logged_in,
            "email": self.email,
            "plan": self.plan,
            "method": self.method,
            "ready": self.ready,
            "description": self.describe(),
        }


@dataclass
class ClaudeAnswer:
    data: dict[str, Any]
    model: str | None = None
    duration_s: float | None = None
    cost_usd: float | None = None  # list-price equivalent; subscriptions aren't billed per call


Runner = Callable[..., subprocess.CompletedProcess]


class ClaudeCLI:
    def __init__(
        self,
        command: str = "claude",
        model: str = "sonnet",
        *,
        timeout: float = DEFAULT_TIMEOUT,
        runner: Runner = subprocess.run,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.argv = shlex.split(command) or ["claude"]
        self.model = model
        self.timeout = timeout
        self._run = runner
        self.env = account_env(environ)  # never API keys: Claude account only

    def available(self) -> bool:
        return shutil.which(self.argv[0]) is not None

    def status(self, timeout: float = 20) -> ClaudeStatus:
        """Sign-in status from `claude auth status --json` (fast; uses no Claude usage)."""
        if not self.available():
            return ClaudeStatus(installed=False)
        try:
            completed = self._run(
                [*self.argv, "auth", "status", "--json"],
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
                env=self.env,
            )
        except (subprocess.TimeoutExpired, OSError):
            return ClaudeStatus(installed=True)
        try:
            data = json.loads(completed.stdout)  # printed even when signed out (exit code 1)
        except (TypeError, json.JSONDecodeError):
            return ClaudeStatus(installed=True)
        if not isinstance(data, dict) or "loggedIn" not in data:
            return ClaudeStatus(installed=True)
        return ClaudeStatus(
            installed=True,
            logged_in=bool(data["loggedIn"]),
            email=data.get("email") or None,
            plan=data.get("subscriptionType") or None,
            method=data.get("authMethod") or None,
        )

    def login_command(self) -> list[str]:
        """Opens the Claude account sign-in page in the browser."""
        return [*self.argv, "auth", "login", "--claudeai"]

    def command(self, system: str, schema: dict[str, Any]) -> list[str]:
        return [
            *self.argv,
            "-p",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--tools",
            "",
            "--no-session-persistence",
            "--setting-sources",
            "",
            "--system-prompt",
            system,
            "--json-schema",
            json.dumps(schema),
        ]

    def ask(self, system: str, prompt: str, schema: dict[str, Any]) -> ClaudeAnswer:
        """Send `prompt` (on stdin) and return the structured answer."""
        if not self.available():
            raise ClaudeError(
                f"`{self.argv[0]}` was not found; install Claude Code (https://claude.com/claude-code)"
            )
        with tempfile.TemporaryDirectory(prefix="arxiv-digest-") as empty:
            try:
                completed = self._run(
                    self.command(system, schema),
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=empty,
                    env=self.env,
                )
            except subprocess.TimeoutExpired as exc:
                raise ClaudeError(f"Claude did not answer within {self.timeout:g} s") from exc
            except OSError as exc:
                raise ClaudeError(f"could not run `{self.argv[0]}`: {exc}") from exc

        try:
            envelope = json.loads(completed.stdout) if completed.stdout.strip() else None
        except json.JSONDecodeError:
            envelope = None
        if not isinstance(envelope, dict):
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            reason = detail[-1] if detail else f"exit code {completed.returncode}"
            raise ClaudeError(f"Claude Code failed: {reason[:300]}")
        if envelope.get("is_error") or envelope.get("subtype") != "success":
            reason = str(envelope.get("result") or envelope.get("subtype") or "unknown error")
            if _NOT_SIGNED_IN.search(reason):
                raise ClaudeNotSignedIn(
                    f"Claude Code isn't signed in to a Claude account ({reason[:120]})"
                )
            raise ClaudeError(f"Claude Code reported an error: {reason[:300]}")

        data = envelope.get("structured_output")
        if not isinstance(data, dict):
            # Older CLI versions put the JSON text in `result` instead.
            try:
                data = json.loads(envelope.get("result") or "")
            except (TypeError, json.JSONDecodeError) as exc:
                raise ClaudeError("Claude's answer was not the expected JSON") from exc
            if not isinstance(data, dict):
                raise ClaudeError("Claude's answer was not the expected JSON")

        usage = envelope.get("modelUsage")
        model = next(iter(usage), None) if isinstance(usage, dict) and usage else None
        duration = envelope.get("duration_ms")
        return ClaudeAnswer(
            data=data,
            model=model,
            duration_s=duration / 1000 if isinstance(duration, (int, float)) else None,
            cost_usd=envelope.get("total_cost_usd"),
        )
