"""`arxiv-digest setup`: what the install scripts run after installing the packages."""

import pytest
from typer.testing import CliRunner

from arxiv_digest import cli, launcher
from arxiv_digest.claude_cli import ClaudeStatus

CONFIG = """
categories = ["cs.LG"]
output_dir = "~/papers"

[[interest]]
name = "x"
keywords = ["y"]
"""


@pytest.fixture
def calls(tmp_path, monkeypatch):
    """Record what setup does instead of touching the system."""
    done: dict = {"opened": False}
    monkeypatch.setenv("ARXIV_DIGEST_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("ARXIV_DIGEST_HOME", str(tmp_path / "home"))

    def install(config=None):
        done["shortcut"] = config
        return []

    def open_app(port, config, **kwargs):
        done["opened"] = True
        return f"http://127.0.0.1:{port}/", True

    monkeypatch.setattr(launcher, "install_shortcut", install)
    monkeypatch.setattr(launcher, "open_app", open_app)
    monkeypatch.setattr(launcher, "server_status", lambda port: done.get("server", "free"))
    monkeypatch.setattr(launcher, "shortcut_place", lambda: "your app launcher")
    status = ClaudeStatus(installed=True, logged_in=True, email="a@b.c", method="claude.ai")
    done["claude"] = status

    class FakeCLI:
        def __init__(self, *args, **kwargs):
            pass

        def status(self):
            return done["claude"]

    monkeypatch.setattr(cli, "ClaudeCLI", FakeCLI)
    return done


def _setup(*args):
    result = CliRunner().invoke(cli.app, ["setup", *args])
    assert result.exit_code == 0, result.output
    return result.output


def test_setup_adds_the_app_and_opens_it(calls, tmp_path):
    output = _setup()
    assert "Added arxiv-digest to your app launcher" in output
    assert "choose your arXiv categories and interests" in output  # no config yet
    assert "Signed in to Claude as a@b.c" in output
    assert calls["opened"]
    assert calls["shortcut"] == (tmp_path / "config.toml").resolve()


def test_setup_with_a_config_and_no_claude(calls, tmp_path):
    (tmp_path / "config.toml").write_text(CONFIG)
    calls["claude"] = ClaudeStatus(installed=False)
    output = _setup("--no-open")
    assert "choose your arXiv categories" not in output  # already set up
    assert "Claude Code isn't installed" in output and "still works" in output
    assert not calls["opened"]


def test_setup_warns_when_an_older_app_is_running(calls):
    calls["server"] = "ours"
    calls["claude"] = ClaudeStatus(installed=True, logged_in=False)
    output = _setup("--no-open")
    assert "already running" in output and "press Quit" in output
    assert "Log in button" in output
