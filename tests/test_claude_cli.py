import json
import os
import subprocess
import sys

import pytest

from arxiv_digest.claude_cli import ClaudeCLI, ClaudeError, ClaudeNotSignedIn

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


def test_not_signed_in_error_is_recognised():
    runner = FakeRunner(stdout=_envelope(is_error=True, result="Not logged in · Please run /login"))
    with pytest.raises(ClaudeNotSignedIn):
        _cli(runner).ask("s", "p", SCHEMA)


def test_status_signed_in():
    out = json.dumps(
        {"loggedIn": True, "authMethod": "claude.ai", "email": "a@b.c", "subscriptionType": "max"}
    )
    runner = FakeRunner(stdout=out)
    status = _cli(runner).status()
    assert (status.installed, status.logged_in, status.email, status.plan) == (
        True,
        True,
        "a@b.c",
        "max",
    )
    assert status.describe() == "Signed in to Claude as a@b.c (Max plan)"
    argv, kwargs = runner.calls[0]
    assert argv[1:] == ["auth", "status", "--json"] and kwargs["stdin"] == subprocess.DEVNULL


def test_status_signed_out_still_parses_output_despite_exit_code():
    runner = FakeRunner(stdout=json.dumps({"loggedIn": False, "authMethod": "none"}), returncode=1)
    status = _cli(runner).status()
    assert status.logged_in is False
    assert status.describe() == "Claude Code is installed but not signed in to a Claude account"
    assert status.ready is False


def test_status_unknown_or_missing():
    assert _cli(FakeRunner(stdout="Usage: claude ...")).status().logged_in is None
    assert _cli(FakeRunner(exc=subprocess.TimeoutExpired("c", 1))).status().logged_in is None
    missing = ClaudeCLI("definitely-not-a-real-command-xyz", runner=FakeRunner())
    assert missing.status().installed is False
    assert missing.login_command()[-3:] == ["auth", "login", "--claudeai"]


def test_api_keys_and_other_providers_are_never_passed_to_claude_code():
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/home/me",
        "ANTHROPIC_API_KEY": "sk-ant-should-not-be-used",
        "ANTHROPIC_AUTH_TOKEN": "tok",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "CLAUDE_CODE_USE_VERTEX": "1",
    }
    runner = FakeRunner(stdout=_envelope())
    cli = ClaudeCLI(sys.executable, runner=runner, environ=environ)
    cli.ask("s", "p", SCHEMA)
    cli.status()
    for _, kwargs in runner.calls:
        env = kwargs["env"]
        assert "/usr/bin" in env["PATH"].split(os.pathsep) and env["HOME"] == "/home/me"
        for name in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_VERTEX",
        ):
            assert name not in env


def test_only_a_claude_account_counts_as_ready():
    console = json.dumps({"loggedIn": True, "authMethod": "console", "email": "a@b.c"})
    status = _cli(FakeRunner(stdout=console)).status()
    assert status.logged_in is True and status.ready is False
    assert "Anthropic Console (API billing)" in status.describe()
    account = json.dumps({"loggedIn": True, "authMethod": "claude.ai"})
    assert _cli(FakeRunner(stdout=account)).status().ready is True


def test_credit_balance_errors_mean_no_claude_account():
    runner = FakeRunner(stdout=_envelope(is_error=True, result="Credit balance is too low"))
    with pytest.raises(ClaudeNotSignedIn, match="Claude account"):
        _cli(runner).ask("s", "p", SCHEMA)


def _fake_claude(folder):
    folder.mkdir(parents=True, exist_ok=True)
    exe = folder / "claude"
    exe.write_text("#!/bin/sh\necho claude\n")
    exe.chmod(0o755)
    return exe


def test_finds_claude_outside_path_like_a_macos_app(tmp_path, monkeypatch):
    from arxiv_digest.claude_cli import find_claude

    find_claude.cache_clear()
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = _fake_claude(tmp_path / ".local" / "bin")
    minimal = "/usr/bin:/bin:/usr/sbin:/sbin"  # what macOS gives apps started from Finder
    assert find_claude("claude", minimal, None) == str(exe)
    cli = ClaudeCLI("claude", environ={"PATH": minimal, "HOME": str(tmp_path)})
    assert cli.available() and cli.argv[0] == str(exe)
    assert cli.env["PATH"].split(os.pathsep)[0] == str(exe.parent)  # node etc. findable too
    find_claude.cache_clear()


def test_asks_the_login_shell_as_a_last_resort(tmp_path, monkeypatch):
    from arxiv_digest.claude_cli import find_claude

    find_claude.cache_clear()
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    exe = _fake_claude(tmp_path / "somewhere" / "odd")
    shell = tmp_path / "fake-shell"
    shell.write_text(f"#!/bin/sh\necho 'welcome banner'\necho {exe}\n")
    shell.chmod(0o755)
    assert find_claude("claude", "/usr/bin:/bin", str(shell)) == str(exe)
    find_claude.cache_clear()


def test_custom_path_and_not_installed(tmp_path, monkeypatch):
    from arxiv_digest.claude_cli import find_claude

    find_claude.cache_clear()
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = _fake_claude(tmp_path / "tools")
    assert find_claude(str(exe), "/usr/bin", None) == str(exe)
    assert find_claude("claude", "/nonexistent", None) is None
    assert find_claude("not-claude-xyz", "/nonexistent", None) is None
    find_claude.cache_clear()
