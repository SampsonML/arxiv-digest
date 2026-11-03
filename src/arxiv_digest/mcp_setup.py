"""Registering the MCP server with Claude Code and Claude Desktop.

    Claude Code     `claude mcp add --scope user arxiv-digest -- <python> -m arxiv_digest mcp serve`
                    (available in every project)
    Claude Desktop  an "arxiv-digest" entry in claude_desktop_config.json
                    (macOS and Windows; a backup of the file is kept)

Both run the Python that `arxiv-digest mcp install` was run with, like the
launcher shortcut.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .claude_cli import ClaudeCLI
from .launcher import python_command

SERVER_NAME = "arxiv-digest"
CARRIED_ENV = ("ARXIV_DIGEST_HOME", "ARXIV_DIGEST_CONFIG")


def serve_command(config: Path | None = None) -> list[str]:
    command = [*python_command(), "mcp", "serve"]
    if config:
        command += ["--config", str(config)]
    return command


def server_env() -> dict[str, str]:
    return {k: v for k in CARRIED_ENV if (v := os.environ.get(k))}


def claude_code_add(claude: str, config: Path | None = None) -> list[str]:
    env = [arg for k, v in server_env().items() for arg in ("-e", f"{k}={v}")]
    return [
        claude,
        "mcp",
        "add",
        "--scope",
        "user",
        SERVER_NAME,
        *env,
        "--",
        *serve_command(config),
    ]


def claude_code_remove(claude: str) -> list[str]:
    return [claude, "mcp", "remove", "--scope", "user", SERVER_NAME]


def desktop_config_path() -> Path | None:
    """Claude Desktop's config file, if Claude Desktop is installed (macOS, Windows)."""
    if sys.platform == "darwin":
        folder = Path.home() / "Library" / "Application Support" / "Claude"
    elif sys.platform == "win32":
        folder = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Claude"
    else:
        return None  # there's no Claude Desktop for Linux
    return folder / "claude_desktop_config.json" if folder.is_dir() else None


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(data, dict):
        raise ValueError(f"{path} isn't a JSON object")
    return data


def _write(path: Path, data: dict) -> None:
    if path.exists():
        shutil.copy2(path, path.with_name(f"{path.name}.bak.{int(time.time())}"))
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def add_to_desktop(path: Path, config: Path | None = None) -> None:
    data = _read(path)
    servers = data.setdefault("mcpServers", {})
    command = serve_command(config)
    entry: dict = {"command": command[0], "args": command[1:]}
    if env := server_env():
        entry["env"] = env
    servers[SERVER_NAME] = entry
    _write(path, data)


def remove_from_desktop(path: Path) -> bool:
    data = _read(path)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or SERVER_NAME not in servers:
        return False
    del servers[SERVER_NAME]
    _write(path, data)
    return True


def _claude() -> ClaudeCLI | None:
    cli = ClaudeCLI()
    return cli if cli.available() else None


def install(config: Path | None = None) -> list[str]:
    """Register with Claude Code and Claude Desktop; returns what was done."""
    done: list[str] = []
    cli = _claude()
    if cli is None:
        done.append("Claude Code isn't installed, so it was skipped.")
    else:
        claude = cli.argv[0]
        # Replace an older registration (e.g. from another Python environment).
        subprocess.run(claude_code_remove(claude), env=cli.env, capture_output=True, check=False)
        result = subprocess.run(
            claude_code_add(claude, config),
            env=cli.env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            done.append("Claude Code: added for all your projects (user scope).")
        else:
            detail = (result.stderr or result.stdout).strip().splitlines()
            done.append(
                "Claude Code: couldn't add it: " + (detail[-1] if detail else "unknown error")
            )
    desktop = desktop_config_path()
    if desktop is not None:
        add_to_desktop(desktop, config)
        done.append(f"Claude Desktop: added to {desktop}. Restart Claude Desktop to use it.")
    return done


def uninstall() -> list[str]:
    done: list[str] = []
    cli = _claude()
    if cli is not None:
        result = subprocess.run(
            claude_code_remove(cli.argv[0]),
            env=cli.env,
            capture_output=True,
            text=True,
            check=False,
        )
        done.append(
            "Claude Code: removed."
            if result.returncode == 0
            else "Claude Code: it wasn't registered."
        )
    desktop = desktop_config_path()
    if desktop is not None and remove_from_desktop(desktop):
        done.append(f"Claude Desktop: removed from {desktop}. Restart Claude Desktop.")
    return done
