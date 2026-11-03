"""Opening the web app without a terminal.

`arxiv-digest open` starts the app in the background if it isn't running yet
(then it stops itself after a while with no open tab) and opens it in the
browser. `arxiv-digest shortcut install` adds an "arxiv-digest" entry with the
test-tube icon to the system's app launcher that runs exactly that:

    Linux    ~/.local/share/applications/arxiv-digest.desktop (+ icons)
    macOS    ~/Applications/arxiv-digest.app
    Windows  Start menu › arxiv-digest

The entry runs the Python that `shortcut install` was run with, so it works
for `uv tool install`, `pipx`, a virtualenv or a clone with `uv run` alike.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import time
import webbrowser
from collections.abc import Callable
from importlib import resources
from pathlib import Path

import httpx

from . import __version__, paths

APP_ID = "arxiv-digest"
DEFAULT_PORT = 8765
IDLE_EXIT = 600  # seconds without an open app tab before a background app stops


class LauncherError(Exception):
    """The app couldn't be started or the launcher entry couldn't be written."""


def app_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/"


def server_status(port: int, timeout: float = 1.0) -> str:
    """ "ours" if arxiv-digest is serving on the port, "free" if nothing is, else "other"."""
    try:
        response = httpx.get(app_url(port) + "health", timeout=timeout)
    except httpx.ConnectError:
        return "free"
    except httpx.HTTPError:
        return "other"
    try:
        return "ours" if response.json().get("app") == APP_ID else "other"
    except ValueError:
        return "other"


def python_command() -> list[str]:
    """How to run arxiv-digest with the current Python (no console window on Windows)."""
    exe = Path(sys.executable)  # not resolved: a virtualenv's python must stay a venv path
    if os.name == "nt":
        pythonw = exe.with_name("pythonw.exe")
        if pythonw.exists():
            exe = pythonw
    return [str(exe), "-m", "arxiv_digest"]


def _extra_env() -> dict[str, str]:
    """Settings from the environment that a launcher entry must carry along."""
    return {k: v for k in ("ARXIV_DIGEST_HOME",) if (v := os.environ.get(k))}


def start_in_background(
    port: int, config: Path | None, idle_exit: float, log_path: Path
) -> subprocess.Popen:
    command = [*python_command(), "app", "--port", str(port), "--no-browser"]
    command += ["--idle-exit", str(int(idle_exit))]
    if config:
        command += ["--config", str(config)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")  # noqa: SIM115 - handed to the child process
    options: dict = {"stdin": subprocess.DEVNULL, "stdout": log, "stderr": subprocess.STDOUT}
    if os.name == "nt":
        options["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        options["start_new_session"] = True  # keeps running after the launcher exits
    try:
        return subprocess.Popen(command, **options)
    finally:
        log.close()


def open_app(
    port: int = DEFAULT_PORT,
    config: Path | None = None,
    *,
    browser: bool = True,
    idle_exit: float = IDLE_EXIT,
    wait: float = 20.0,
    status: Callable[[int], str] = server_status,
    start: Callable[..., object] = start_in_background,
    open_url: Callable[[str], object] = webbrowser.open,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, bool]:
    """Make sure the app is running, then open it. Returns (url, started_now)."""
    url = app_url(port)
    state = status(port)
    if state == "other":
        raise LauncherError(f"Port {port} is used by another program; try --port {port + 1}.")
    started = state == "free"
    if started:
        log_path = paths.data_dir() / "app.log"
        start(port, config, idle_exit, log_path)
        deadline = time.monotonic() + wait
        while status(port) != "ours":
            if time.monotonic() > deadline:
                raise LauncherError(f"The app didn't start; see {log_path}")
            sleep(0.25)
    if browser:
        open_url(url)
    return url, started


# ---------------------------------------------------------------------------
# Launcher entries


def launch_command(config: Path | None = None) -> list[str]:
    command = [*python_command(), "open"]
    if config:
        command += ["--config", str(config)]
    return command


def _icon(name: str) -> bytes:
    return resources.files("arxiv_digest").joinpath("assets", "icons", name).read_bytes()


_DESKTOP_RESERVED = set(" \t\n\"'\\><~|&;$*?#()`")


def _desktop_quote(arg: str) -> str:
    """Quote one Exec argument as the Desktop Entry specification requires."""
    arg = arg.replace("%", "%%")
    if not any(ch in _DESKTOP_RESERVED for ch in arg):
        return arg
    inner = arg.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
    return f'"{inner}"'.replace("\\", "\\\\")  # Exec is also a string value: escape again


def desktop_entry(command: list[str], env: dict[str, str] | None = None) -> str:
    argv = (["env", *(f"{k}={v}" for k, v in env.items())] if env else []) + command
    return "\n".join(
        [
            "[Desktop Entry]",
            "Type=Application",
            "Version=1.5",
            "Name=arxiv-digest",
            "GenericName=arXiv digest",
            "Comment=Your daily digest of new arXiv papers",
            "Exec=" + " ".join(_desktop_quote(a) for a in argv),
            "TryExec=" + _desktop_quote(command[0]),
            f"Icon={APP_ID}",
            "Terminal=false",
            "Categories=Education;Literature;",
            "Keywords=arxiv;papers;research;digest;science;",
            "StartupNotify=false",
            "",
        ]
    )


def _data_home() -> Path:
    value = os.environ.get("XDG_DATA_HOME")
    return Path(value).expanduser() if value else Path.home() / ".local" / "share"


def _linux_files() -> dict[str, Path]:
    home = _data_home()
    return {
        "entry": home / "applications" / f"{APP_ID}.desktop",
        "svg": home / "icons" / "hicolor" / "scalable" / "apps" / f"{APP_ID}.svg",
        "png": home / "icons" / "hicolor" / "256x256" / "apps" / f"{APP_ID}.png",
    }


def _refresh_linux_caches() -> None:
    home = _data_home()
    for command in (
        ["update-desktop-database", str(home / "applications")],
        ["gtk-update-icon-cache", "-f", "-t", "-q", str(home / "icons" / "hicolor")],
    ):
        if shutil.which(command[0]):
            subprocess.run(command, capture_output=True, check=False)


def _install_linux(command: list[str]) -> list[Path]:
    files = _linux_files()
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    files["svg"].write_bytes(_icon("arxiv-digest.svg"))
    files["png"].write_bytes(_icon("arxiv-digest-256.png"))
    files["entry"].write_text(desktop_entry(command, _extra_env()), encoding="utf-8")
    _refresh_linux_caches()
    return list(files.values())


def _macos_app() -> Path:
    return Path.home() / "Applications" / f"{APP_ID}.app"


def macos_script(command: list[str], env: dict[str, str] | None = None) -> str:
    exports = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in (env or {}).items())
    return "#!/bin/sh\n" + exports + "exec " + " ".join(shlex.quote(a) for a in command) + "\n"


def _install_macos(command: list[str]) -> list[Path]:
    app = _macos_app()
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    (contents / "Resources").mkdir(parents=True, exist_ok=True)
    info = {
        "CFBundleName": APP_ID,
        "CFBundleDisplayName": APP_ID,
        "CFBundleIdentifier": "io.github.arxiv-digest.launcher",
        "CFBundleExecutable": APP_ID,
        "CFBundleIconFile": APP_ID,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "LSUIElement": True,  # just opens the browser; no Dock icon
    }
    (contents / "Info.plist").write_bytes(plistlib.dumps(info))
    script = contents / "MacOS" / APP_ID
    # Apps opened from Finder/Spotlight get a minimal PATH; keep the terminal's, so
    # Claude Code (often in ~/.local/bin or Homebrew) is found.
    env = {**_extra_env(), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    script.write_text(macos_script(command, env), encoding="utf-8")
    script.chmod(0o755)
    (contents / "Resources" / f"{APP_ID}.icns").write_bytes(_icon("arxiv-digest.icns"))
    return [app]


def _windows_shortcut() -> Path:
    appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / f"{APP_ID}.lnk"


WINDOWS_SCRIPT = (
    "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:AD_LINK); "
    "$s.TargetPath = $env:AD_TARGET; $s.Arguments = $env:AD_ARGS; "
    "$s.IconLocation = $env:AD_ICON; $s.Description = 'Your daily arXiv digest'; $s.Save()"
)


def _install_windows(command: list[str]) -> list[Path]:
    link = _windows_shortcut()
    icon = paths.data_dir() / f"{APP_ID}.ico"
    icon.parent.mkdir(parents=True, exist_ok=True)
    icon.write_bytes(_icon("arxiv-digest.ico"))
    link.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "AD_LINK": str(link),
        "AD_TARGET": command[0],
        "AD_ARGS": subprocess.list2cmdline(command[1:]),
        "AD_ICON": str(icon),
    }
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", WINDOWS_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise LauncherError(f"Couldn't create the Start menu shortcut: {result.stderr.strip()}")
    return [link, icon]


def install_shortcut(config: Path | None = None) -> list[Path]:
    command = launch_command(config)
    if sys.platform == "darwin":
        return _install_macos(command)
    if sys.platform == "win32":
        return _install_windows(command)
    return _install_linux(command)


def remove_shortcut() -> list[Path]:
    if sys.platform == "darwin":
        targets = [_macos_app()]
    elif sys.platform == "win32":
        targets = [_windows_shortcut(), paths.data_dir() / f"{APP_ID}.ico"]
    else:
        targets = list(_linux_files().values())
    removed = []
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(target)
        elif target.exists():
            target.unlink()
            removed.append(target)
    if sys.platform not in ("darwin", "win32"):
        _refresh_linux_caches()
    return removed


def shortcut_installed() -> bool:
    if sys.platform == "darwin":
        return _macos_app().exists()
    if sys.platform == "win32":
        return _windows_shortcut().exists()
    return _linux_files()["entry"].exists()


def shortcut_place() -> str:
    """Where the launcher entry goes, in words for the web app's offer."""
    if sys.platform == "darwin":
        return "Applications"
    if sys.platform == "win32":
        return "the Start menu"
    return "your app launcher"


def _offer_flag() -> Path:
    return paths.data_dir() / "shortcut-offer-dismissed"


def offer_dismissed() -> bool:
    return _offer_flag().exists()


def dismiss_offer() -> None:
    flag = _offer_flag()
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("The web app no longer offers to add a launcher entry.\n", encoding="utf-8")
