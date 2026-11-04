"""Where arxiv-digest keeps its files.

The XDG base-directory layout is used on every platform (macOS included), so the
locations are the same everywhere and easy to document:

    config  $XDG_CONFIG_HOME/arxiv-digest/config.toml   (default ~/.config/...)
    data    $XDG_DATA_HOME/arxiv-digest/                (default ~/.local/share/...)
    cache   $XDG_CACHE_HOME/arxiv-digest/               (default ~/.cache/...)
            models/ (local embedding model), voices/ and audio/ (Listen)

Overrides: ARXIV_DIGEST_CONFIG points at a different config file, and
ARXIV_DIGEST_HOME puts the data and cache directories under one folder (useful
for testing or for keeping several independent digests).

Digests are kept in the database; files are written only when asked for (the
app's Save button downloads one, and `arxiv-digest run --save` writes Markdown
and HTML to `output_dir`). The default `output_dir` is the `digests/` folder of
the project when arxiv-digest runs from a clone of the repository (`uv run`, or
an editable install); an installed copy (pip, uv tool, uvx) has no project
folder and uses ~/arxiv-digest instead.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

APP = "arxiv-digest"


def _xdg(variable: str, default: str) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser() if value else Path.home() / default


def _home_override() -> Path | None:
    value = os.environ.get("ARXIV_DIGEST_HOME")
    return Path(value).expanduser() if value else None


def config_path() -> Path:
    override = os.environ.get("ARXIV_DIGEST_CONFIG")
    if override:
        return Path(override).expanduser()
    return _xdg("XDG_CONFIG_HOME", ".config") / APP / "config.toml"


def data_dir() -> Path:
    home = _home_override()
    return home / "data" if home else _xdg("XDG_DATA_HOME", ".local/share") / APP


def cache_dir() -> Path:
    home = _home_override()
    return home / "cache" if home else _xdg("XDG_CACHE_HOME", ".cache") / APP


def database_path() -> Path:
    return data_dir() / "arxiv-digest.sqlite3"


def model_cache_dir() -> Path:
    return cache_dir() / "models"


def voices_dir() -> Path:
    """Piper voices for Listen (audio.py), downloaded on first use."""
    return cache_dir() / "voices"


def audio_dir() -> Path:
    """Spoken digests (MP3), so a digest is only spoken once."""
    return cache_dir() / "audio"


def source_checkout(package_dir: Path | None = None) -> Path | None:
    """The project folder when running from a clone of the repository, else None.

    A clone has the src layout: <project>/src/arxiv_digest/ next to a
    <project>/pyproject.toml that names this package.
    """
    package_dir = (package_dir or Path(__file__).parent).resolve()
    if package_dir.parent.name != "src":
        return None
    project = package_dir.parent.parent
    try:
        text = (project / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    if re.search(r'^name\s*=\s*"arxiv-digest"\s*$', text, re.M) is None:
        return None
    return project


def display(path: Path) -> str:
    """A path written with ~ for the home folder, as it would go in a config file."""
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(path)


def default_output_dir() -> str:
    """Where digests go when the config doesn't say (see the module docstring)."""
    project = source_checkout()
    return display(project / "digests") if project else "~/arxiv-digest"
