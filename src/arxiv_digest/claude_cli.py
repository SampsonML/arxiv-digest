"""Asking Claude questions through the Claude Code CLI (`claude -p`).

This uses the person's own Claude Code sign-in (e.g. a Claude subscription), so
no API key is needed. Each call is isolated from their usual Claude Code setup:

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
    ) -> None:
        self.argv = shlex.split(command) or ["claude"]
        self.model = model
        self.timeout = timeout
        self._run = runner

    def available(self) -> bool:
        return shutil.which(self.argv[0]) is not None

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
            reason = envelope.get("result") or envelope.get("subtype") or "unknown error"
            raise ClaudeError(f"Claude Code reported an error: {str(reason)[:300]}")

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
