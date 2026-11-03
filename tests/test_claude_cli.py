import json
import os
import subprocess
import sys

import pytest

from arxiv_digest.claude_cli import ClaudeCLI, ClaudeError

SCHEMA = {"type": "object"}


class FakeRunner:
    def __init__(self, stdout="", stderr="", returncode=0, exc=None):
        self.result = subprocess.CompletedProcess([], returncode, stdout, stderr)
        self.exc = exc
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.exc:
            raise self.exc
        return self.result


def _envelope(**overrides):
    data = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": '{"picks": []}',
        "structured_output": {"picks": [{"id": "1"}]},
        "duration_ms": 2500,
        "total_cost_usd": 0.01,
        "modelUsage": {"claude-sonnet-test": {}},
    }
    data.update(overrides)
    return json.dumps(data)


def _cli(runner):
    # Any command that exists on PATH; the fake runner never executes it.
    return ClaudeCLI(sys.executable, "sonnet", runner=runner)


def test_ask_runs_isolated_and_returns_structured_output():
    runner = FakeRunner(stdout=_envelope())
    answer = _cli(runner).ask("SYSTEM", "PROMPT", SCHEMA)
    assert answer.data == {"picks": [{"id": "1"}]}
    assert answer.model == "claude-sonnet-test"
    assert answer.duration_s == 2.5
    argv, kwargs = runner.calls[0]
    assert argv[0] == sys.executable and argv[1] == "-p"
    joined = " ".join(f"[{a}]" for a in argv)
    assert "[--tools] []" in joined  # all tools disabled
    assert "[--setting-sources] []" in joined  # user settings and hooks not loaded
    assert "[--no-session-persistence]" in joined
    assert "[--system-prompt] [SYSTEM]" in joined
    assert argv[argv.index("--json-schema") + 1] == json.dumps(SCHEMA)
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert kwargs["input"] == "PROMPT"
    assert kwargs["cwd"] != os.getcwd() and not os.path.exists(kwargs["cwd"])  # temp, cleaned up


def test_ask_falls_back_to_json_in_result_text():
    runner = FakeRunner(stdout=_envelope(structured_output=None, result='{"picks": []}'))
    assert _cli(runner).ask("s", "p", SCHEMA).data == {"picks": []}


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        (FakeRunner(stdout=_envelope(is_error=True, result="usage limit reached")), "usage limit"),
        (FakeRunner(stdout="", stderr="boom\nNot logged in", returncode=1), "Not logged in"),
        (FakeRunner(stdout=_envelope(structured_output=None, result="sorry")), "expected JSON"),
        (FakeRunner(exc=subprocess.TimeoutExpired("claude", 5)), "within"),
        (FakeRunner(exc=FileNotFoundError("nope")), "could not run"),
    ],
)
def test_ask_errors_are_explained(runner, message):
    with pytest.raises(ClaudeError, match=message):
        _cli(runner).ask("s", "p", SCHEMA)


def test_missing_command():
    cli = ClaudeCLI("definitely-not-a-real-command-xyz", runner=FakeRunner())
    assert not cli.available()
    with pytest.raises(ClaudeError, match="not found"):
        cli.ask("s", "p", SCHEMA)
