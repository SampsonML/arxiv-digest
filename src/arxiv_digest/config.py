"""Reading, validating, creating and editing config.toml."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomlkit
from tomlkit.exceptions import TOMLKitError

from . import categories as cats
from . import paths

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEEP_READ_MODEL = "sonnet"  # deep reads use this unless deep_read_model is set

TOP_LEVEL_KEYS = {
    "categories",
    "output_dir",
    "max_papers",
    "top_picks",
    "shortlist_size",
    "max_catchup_days",
    "embedding_model",
    "min_zscore",
    "allow_unknown_categories",
    "use_claude",
    "claude_model",
    "claude_command",
    "claude_timeout",
    "deep_reads",
    "deep_read_model",
    "app_background",
    "interest",
}
INTEREST_KEYS = {"name", "description", "keywords", "require", "exclude", "authors"}


class ConfigError(Exception):
    """A problem with config.toml, explained so the user can fix it."""


@dataclass
class Interest:
    name: str
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    require: list[str] = field(default_factory=list)  # every term must appear
    exclude: list[str] = field(default_factory=list)  # any term rules the paper out
    authors: list[str] = field(default_factory=list)  # always include their papers

    @property
    def has_topic(self) -> bool:
        """Whether the interest describes a topic (so it can be matched by meaning)."""
        return bool(self.description.strip() or self.keywords)

    @property
    def query_text(self) -> str:
        """The text the local model compares papers against."""
        parts = [self.name.strip()]
        if self.description.strip():
            parts.append(self.description.strip())
        if self.keywords:
            parts.append("Keywords: " + ", ".join(self.keywords))
        return ". ".join(parts)


@dataclass
class Config:
    path: Path
    categories: list[str]
    interests: list[Interest]
    output_dir: Path
    max_papers: int = 15
    top_picks: int = 3
    shortlist_size: int = 40
    max_catchup_days: float = 7.0
    embedding_model: str = DEFAULT_MODEL
    min_zscore: float = 2.5
    allow_unknown_categories: bool = False
    use_claude: bool = True
    claude_model: str = "sonnet"
    claude_command: str = "claude"
    claude_timeout: float = 600.0
    deep_reads: int = 3  # top picks Claude reads in full (0: none)
    deep_read_model: str = DEEP_READ_MODEL
    app_background: str = "default"  # "default", "none", or a path to an image


def _expect(cond: bool, message: str) -> None:
    if not cond:
        raise ConfigError(message)


def _str_list(value: Any, where: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    _expect(
        isinstance(value, list) and all(isinstance(v, str) for v in value),
        f'{where} must be a list of strings, e.g. ["one", "two"]',
    )
    return [v.strip() for v in value if v.strip()]


def _number(data: dict[str, Any], key: str, default: float, low: float, high: float) -> float:
    value = data.get(key, default)
    _expect(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"`{key}` must be a number",
    )
    _expect(low <= value <= high, f"`{key}` must be between {low:g} and {high:g} (got {value})")
    return value


def _resolve_dir(value: str, config_path: Path) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def parse(text: str, path: Path) -> Config:
    try:
        data = tomlkit.parse(text).unwrap()
    except TOMLKitError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc

    unknown = set(data) - TOP_LEVEL_KEYS
    _expect(
        not unknown,
        f"Unknown setting(s) in {path}: {', '.join(sorted(unknown))}. "
        f"Known settings: {', '.join(sorted(TOP_LEVEL_KEYS - {'interest'}))}, [[interest]].",
    )

    allow_unknown = bool(data.get("allow_unknown_categories", False))
    raw_categories = _str_list(data.get("categories", []), "`categories`")
    _expect(bool(raw_categories), '`categories` is empty: list at least one, e.g. ["cs.LG"]')
    categories: list[str] = []
    for code in raw_categories:
        official = cats.canonical(code)
        if official is None:
            hint = cats.suggestions(code)
            _expect(
                allow_unknown,
                f"Unknown arXiv category {code!r}."
                + (f" Did you mean {', '.join(hint)}?" if hint else "")
                + " See `arxiv-digest categories list --all`. If arXiv has added it"
                " recently, set allow_unknown_categories = true.",
            )
            official = code.strip()
        if official not in categories:
            categories.append(official)

    raw_interests = data.get("interest", [])
    _expect(isinstance(raw_interests, list), "Interests must be written as [[interest]] tables")
    _expect(bool(raw_interests), "No interests defined: add at least one [[interest]] table")
    interests: list[Interest] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_interests, start=1):
        where = f"[[interest]] #{index}"
        _expect(isinstance(raw, dict), f"{where} must be a table")
        unknown = set(raw) - INTEREST_KEYS
        _expect(
            not unknown,
            f"Unknown key(s) in {where}: {', '.join(sorted(unknown))}. "
            f"Known keys: {', '.join(sorted(INTEREST_KEYS))}.",
        )
        name = raw.get("name")
        _expect(isinstance(name, str) and name.strip() != "", f"{where} needs a `name`")
        name = name.strip()
        where = f"interest {name!r}"
        _expect(name.lower() not in names, f"Two interests are called {name!r}; names must differ")
        names.add(name.lower())
        description = raw.get("description", "")
        _expect(isinstance(description, str), f"`description` in {where} must be a string")
        interest = Interest(
            name=name,
            description=description.strip(),
            keywords=_str_list(raw.get("keywords", []), f"`keywords` in {where}"),
            require=_str_list(raw.get("require", []), f"`require` in {where}"),
            exclude=_str_list(raw.get("exclude", []), f"`exclude` in {where}"),
            authors=_str_list(raw.get("authors", []), f"`authors` in {where}"),
        )
        _expect(
            interest.has_topic or interest.authors,
            f"{where} needs a `description`, `keywords` or `authors` to match papers",
        )
        interests.append(interest)

    output_dir = data.get("output_dir", paths.default_output_dir())
    _expect(isinstance(output_dir, str) and output_dir.strip() != "", "`output_dir` must be a path")
    model = data.get("embedding_model", DEFAULT_MODEL)
    _expect(isinstance(model, str) and model.strip() != "", "`embedding_model` must be a name")
    use_claude = data.get("use_claude", True)
    _expect(isinstance(use_claude, bool), "`use_claude` must be true or false")
    claude_model = data.get("claude_model", "sonnet")
    _expect(
        isinstance(claude_model, str) and claude_model.strip() != "",
        '`claude_model` must be a model name or alias, e.g. "sonnet"',
    )
    claude_command = data.get("claude_command", "claude")
    _expect(
        isinstance(claude_command, str) and claude_command.strip() != "",
        "`claude_command` must be the command that runs Claude Code",
    )

    return Config(
        path=path,
        categories=categories,
        interests=interests,
        output_dir=_resolve_dir(output_dir, path),
        max_papers=int(_number(data, "max_papers", 15, 1, 200)),
        top_picks=int(_number(data, "top_picks", 3, 0, 20)),
        shortlist_size=int(_number(data, "shortlist_size", 40, 5, 500)),
        max_catchup_days=float(_number(data, "max_catchup_days", 7, 1, 60)),
        embedding_model=model.strip(),
        min_zscore=float(_number(data, "min_zscore", 2.5, 0, 10)),
        allow_unknown_categories=allow_unknown,
        use_claude=use_claude,
        claude_model=claude_model.strip(),
        claude_command=claude_command.strip(),
        claude_timeout=float(_number(data, "claude_timeout", 600, 30, 3600)),
        deep_reads=int(_number(data, "deep_reads", 3, 0, 10)),
        deep_read_model=_model_name(
            data.get("deep_read_model", DEEP_READ_MODEL), "deep_read_model"
        ),
        app_background=_background(data.get("app_background", "default"), path),
    )


def _model_name(value: Any, key: str) -> str:
    _expect(
        isinstance(value, str) and value.strip() != "",
        f'`{key}` must be a model name or alias, e.g. "sonnet"',
    )
    return value.strip()


BACKGROUND_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _background(value: Any, config_path: Path) -> str:
    _expect(
        isinstance(value, str) and value.strip() != "",
        '`app_background` must be "default", "none" or the path to an image',
    )
    value = value.strip()
    if value in ("default", "none"):
        return value
    path = _resolve_dir(value, config_path)
    _expect(
        path.suffix.lower() in BACKGROUND_TYPES,
        f"`app_background` must be a .jpg, .png or .webp image (got {value!r})",
    )
    return str(path)


def load(path: Path) -> Config:
    if not path.exists():
        raise ConfigError(f"No config at {path}. Create one with `arxiv-digest init`.")
    return parse(path.read_text(encoding="utf-8"), path)


# ---------------------------------------------------------------------------
# Creating a config


def _toml_str(value: str) -> str:
    # JSON string escapes are valid in TOML basic strings.
    return json.dumps(value, ensure_ascii=False)


def _toml_list(values: list[str]) -> str:
    return "[" + ", ".join(_toml_str(v) for v in values) + "]"


def render_template(
    categories: list[str],
    output_dir: str,
    interests: list[Interest],
) -> str:
    lines = [
        "# arxiv-digest configuration. Run `arxiv-digest run` after editing.",
        "",
        "# arXiv categories to watch (papers cross-listed into them are included).",
        "# Codes: `arxiv-digest categories list --all`.",
        f"categories = {_toml_list(categories)}",
        "",
        "# Where the daily Markdown files are written (an Obsidian vault folder works).",
        f"output_dir = {_toml_str(output_dir)}",
        "",
        "# Most papers in one digest, and how many go in the Top picks section.",
        "max_papers = 15",
        "top_picks = 3",
        "",
        "# How many papers the local model shortlists for the final ranking.",
        "shortlist_size = 40",
        "",
        "# The first run covers arXiv's latest announcement. Later runs pick up",
        "# everything new since the previous run, up to max_catchup_days back.",
        "max_catchup_days = 7",
        "",
        "# Without Claude, a paper is picked on meaning alone when it is this many",
        "# standard deviations closer to an interest than the day's average paper",
        "# (2.5 is roughly the closest 1%). Keyword and author matches always count.",
        "min_zscore = 2.5",
        "",
        "# Claude picks the final papers from the shortlist and explains each choice.",
        "# It runs through the Claude Code CLI with your Claude sign-in; set",
        "# use_claude = false (or run with --no-claude) to rank on this computer only.",
        "use_claude = true",
        '# "sonnet" (default), "haiku" (faster, lighter) or "opus" (most thorough).',
        'claude_model = "sonnet"',
        "",
        "# Claude reads the full text of this many top picks and summarises their",
        "# method and results (0 turns it off). Deep reads use Sonnet whatever",
        "# claude_model is; set deep_read_model to change that.",
        "deep_reads = 3",
        '# deep_read_model = "sonnet"',
        "",
        '# Background of `arxiv-digest app`: "default" (a night-sky image),',
        '# "none", or the path to your own .jpg/.png/.webp image.',
        '# app_background = "default"',
        "",
        "# The local embedding model (downloaded once, ~90 MB, runs on the CPU).",
        '# "BAAI/bge-small-en-v1.5" ranks a little better but uses ~4x the CPU.',
        f"# embedding_model = {_toml_str(DEFAULT_MODEL)}",
        "",
        "# One [[interest]] table per topic. Every key except name is optional, but",
        "# each interest needs a description, keywords or authors.",
        "#   description  a sentence or two; matched by meaning, not exact words",
        "#   keywords     words or phrases that make a paper more relevant",
        "#   require      every one of these must appear in the title or abstract",
        "#   exclude      a paper containing any of these is skipped",
        '#   authors      their new papers are always included ("First Last")',
    ]
    for interest in interests:
        lines += ["", "[[interest]]", f"name = {_toml_str(interest.name)}"]
        if interest.description:
            lines.append(f"description = {_toml_str(interest.description)}")
        if interest.keywords:
            lines.append(f"keywords = {_toml_list(interest.keywords)}")
        if interest.require:
            lines.append(f"require = {_toml_list(interest.require)}")
        if interest.exclude:
            lines.append(f"exclude = {_toml_list(interest.exclude)}")
        if interest.authors:
            lines.append(f"authors = {_toml_list(interest.authors)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Editing a config in place (keeps comments and formatting)


def _edit(path: Path) -> tuple[tomlkit.TOMLDocument, Config]:
    current = load(path)  # validate before touching anything
    return tomlkit.parse(path.read_text(encoding="utf-8")), current


def _save(path: Path, doc: tomlkit.TOMLDocument) -> Config:
    text = tomlkit.dumps(doc)
    config = parse(text, path)  # never write a config that wouldn't load
    path.write_text(text, encoding="utf-8")
    return config


def add_categories(path: Path, codes: list[str]) -> tuple[Config, list[str]]:
    """Add categories; returns the new config and the codes actually added."""
    doc, current = _edit(path)
    added = []
    for code in codes:
        official = cats.canonical(code)
        if official is None and not current.allow_unknown_categories:
            hint = cats.suggestions(code)
            raise ConfigError(
                f"Unknown arXiv category {code!r}."
                + (f" Did you mean {', '.join(hint)}?" if hint else "")
            )
        official = official or code.strip()
        if official not in current.categories and official not in added:
            added.append(official)
    if added:
        array = doc["categories"]
        if isinstance(array, list):  # tomlkit arrays are lists; keeps their formatting
            for code in added:
                array.append(code)
        else:  # categories = "cs.LG" written as a single string
            doc["categories"] = current.categories + added
    return _save(path, doc), added


def remove_categories(path: Path, codes: list[str]) -> tuple[Config, list[str]]:
    """Remove categories; returns the new config and the codes actually removed."""
    doc, current = _edit(path)
    wanted = {cats.canonical(c) or c.strip() for c in codes}
    removed = [c for c in current.categories if c in wanted]
    _expect(
        len(removed) < len(current.categories),
        "That would remove every category; at least one must remain",
    )
    array = doc["categories"]
    if isinstance(array, list):
        for index in reversed(range(len(array))):
            value = str(array[index])
            if (cats.canonical(value) or value) in removed:
                del array[index]
    else:
        doc["categories"] = [c for c in current.categories if c not in removed]
    return _save(path, doc), removed


# ---------------------------------------------------------------------------
# Saving everything at once (the web app's settings form)


def _interest_table(interest: Interest) -> tomlkit.items.Table:
    table = tomlkit.table()
    table.add("name", interest.name)
    for key in ("description",):
        if getattr(interest, key):
            table.add(key, getattr(interest, key))
    for key in ("keywords", "require", "exclude", "authors"):
        values = getattr(interest, key)
        if values:
            table.add(key, values)
    return table


def save_settings(
    path: Path,
    *,
    categories: list[str],
    interests: list[Interest],
    values: dict[str, Any],
) -> Config:
    """Write categories, interests and top-level settings, keeping other comments.

    Creates the config if it doesn't exist yet. Nothing is written unless the
    result loads cleanly, so a bad form submission can't break the config.
    """
    if path.exists():
        doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    else:
        output_dir = str(values.get("output_dir") or paths.default_output_dir())
        doc = tomlkit.parse(render_template(categories or ["cs.LG"], output_dir, []))
    unknown = set(values) - (TOP_LEVEL_KEYS - {"categories", "interest"})
    _expect(not unknown, f"Unknown setting(s): {', '.join(sorted(unknown))}")
    doc["categories"] = categories
    for key, value in values.items():
        doc[key] = value
    aot = tomlkit.aot()
    for interest in interests:
        aot.append(_interest_table(interest))
    doc["interest"] = aot
    text = tomlkit.dumps(doc)
    config = parse(text, path)  # raises ConfigError with a clear message
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return config


def save_interests(path: Path, interests: list[Interest]) -> Config:
    """Replace only the [[interest]] tables; everything else in the file is kept."""
    doc, _ = _edit(path)
    aot = tomlkit.aot()
    for interest in interests:
        aot.append(_interest_table(interest))
    doc["interest"] = aot
    return _save(path, doc)
