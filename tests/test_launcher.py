import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from arxiv_digest import launcher


def test_desktop_entry_quotes_paths_and_validates(tmp_path):
    command = ["/home/me/My Projects/.venv/bin/python", "-m", "arxiv_digest", "open"]
    entry = launcher.desktop_entry(command, {"ARXIV_DIGEST_HOME": "/tmp/a b"})
    assert 'Exec=env "ARXIV_DIGEST_HOME=/tmp/a b" "/home/me/My Projects/.venv/bin/python"' in entry
    assert entry.splitlines()[0] == "[Desktop Entry]"
    assert "Icon=arxiv-digest" in entry and "Terminal=false" in entry
    if shutil.which("desktop-file-validate"):
        path = tmp_path / "arxiv-digest.desktop"
        path.write_text(entry)
        result = subprocess.run(
            ["desktop-file-validate", str(path)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("arg", "quoted"),
    [
        ("plain", "plain"),
        ("50%", "50%%"),
        ("has space", '"has space"'),
        ('say "hi"', '"say \\\\"hi\\\\""'),
        ("$HOME", '"\\\\$HOME"'),
    ],
)
def test_desktop_quoting(arg, quoted):
    assert launcher._desktop_quote(arg) == quoted


def test_launch_command_uses_this_python():
    assert launcher.launch_command() == [sys.executable, "-m", "arxiv_digest", "open"]
    assert launcher.launch_command(Path("/c.toml"))[-2:] == ["--config", "/c.toml"]


@pytest.mark.skipif(sys.platform in ("darwin", "win32"), reason="Linux launcher layout")
def test_install_and_remove_on_linux(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    created = launcher.install_shortcut()
    entry = tmp_path / "applications" / "arxiv-digest.desktop"
    svg = tmp_path / "icons" / "hicolor" / "scalable" / "apps" / "arxiv-digest.svg"
    png = tmp_path / "icons" / "hicolor" / "256x256" / "apps" / "arxiv-digest.png"
    assert set(created) == {entry, svg, png}
    assert "-m arxiv_digest open" in entry.read_text()
    assert svg.read_bytes().startswith(b"<svg") and png.read_bytes()[:4] == b"\x89PNG"
    assert set(launcher.remove_shortcut()) == {entry, svg, png}
    assert not entry.exists() and launcher.remove_shortcut() == []


def test_macos_script_quotes_the_command():
    script = launcher.macos_script(["/a b/python", "-m", "arxiv_digest", "open"], {"X": "1 2"})
    assert script == "#!/bin/sh\nexport X='1 2'\nexec '/a b/python' -m arxiv_digest open\n"


def test_icons_are_bundled():
    assert launcher._icon("arxiv-digest.ico")[:4] == b"\x00\x00\x01\x00"
    assert launcher._icon("arxiv-digest.icns")[:4] == b"icns"


def _fake_open(states):
    calls = {"started": [], "opened": []}
    seq = iter(states)

    def status(port):
        return next(seq)

    def start(port, config, idle_exit, log_path):
        calls["started"].append((port, config, idle_exit))

    return calls, dict(
        status=status,
        start=start,
        open_url=calls["opened"].append,
        sleep=lambda s: None,
    )


def test_open_app_starts_a_background_server_when_needed():
    calls, fakes = _fake_open(["free", "free", "ours"])
    url, started = launcher.open_app(8800, Path("/c.toml"), **fakes)
    assert started and url == "http://127.0.0.1:8800/"
    assert calls["started"] == [(8800, Path("/c.toml"), launcher.IDLE_EXIT)]
    assert calls["opened"] == [url]


def test_open_app_reuses_a_running_server():
    calls, fakes = _fake_open(["ours"])
    url, started = launcher.open_app(8800, **fakes)
    assert not started and calls["started"] == [] and calls["opened"] == [url]


def test_open_app_errors():
    _, fakes = _fake_open(["other"])
    with pytest.raises(launcher.LauncherError, match="another program"):
        launcher.open_app(8800, **fakes)
    _, fakes = _fake_open(["free"] + ["free"] * 1000)
    with pytest.raises(launcher.LauncherError, match="didn't start"):
        launcher.open_app(8800, wait=0, **fakes)
