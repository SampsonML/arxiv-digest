"""Search terms from Claude: the jargon the local model doesn't know.

The local embedding model is small, and reads a phrase like "flow matching"
roughly as "flow" + "matching", so it ranks graph matching and optical flow
papers above papers about flow matching. Before ranking, Claude turns each
interest into the exact phrases papers on it use ("flow matching", "rectified
flow", "stochastic interpolants", ...). These count like the interest's own
keywords: a paper containing one gets a keyword bonus and a reserved place in
the shortlist.

One small call per interest, made only when the interest changes; the terms
are cached in the database. Without Claude, cached terms are still used.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Protocol

from .claude_cli import ClaudeCLI, ClaudeError
from .config import Config, Interest

PROMPT_VERSION = 1
MAX_TERMS = 10
_SPACE = re.compile(r"\s+")

SYSTEM_PROMPT = """\
You help find new arXiv papers on a researcher's interest by exact text \
matching. Given the interest, list the phrases that papers on this topic \
actually use in their titles and abstracts.

Return 4 to 10 terms:
- the established name of the topic, its common variants and synonyms, and \
the names of closely related methods or problems that someone with this \
interest would want to see;
- each term a short phrase of 1 to 4 words, written as it would appear in an \
abstract (no quotes, no wildcards, no boolean operators);
- specific enough that a paper containing it is very likely on the topic. Never \
a single generic word ("flow", "matching", "learning", "model", "galaxy") or a \
broad field name. Acronyms only if they are unambiguous in the field.

Don't repeat the researcher's own keywords, and don't drift to other topics."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "terms": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_TERMS},
    },
    "required": ["terms"],
    "additionalProperties": False,
}


class Cache(Protocol):
    """Where the terms are kept between runs (the Store)."""

    def get_expansion(self, key: str) -> list[str] | None: ...

    def put_expansion(self, key: str, terms: Sequence[str]) -> None: ...


def cache_key(model: str, interest: Interest) -> str:
    parts = [str(PROMPT_VERSION), model, interest.name, interest.description.strip()]
    parts.append(",".join(interest.keywords))
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def build_prompt(interest: Interest) -> str:
    lines = [f"Interest: {interest.name}"]
    if interest.description.strip():
        lines.append(f"Description: {interest.description.strip()}")
    if interest.keywords:
        lines.append("The researcher's own keywords: " + ", ".join(interest.keywords))
    lines += ["", "List the search terms as instructed."]
    return "\n".join(lines)


def clean_terms(values: Any, interest: Interest) -> list[str]:
    """Claude's terms, tidied: no duplicates, no repeats of the user's keywords."""
    seen = {k.casefold() for k in interest.keywords}
    terms: list[str] = []
    for value in values if isinstance(values, list) else []:
        if not isinstance(value, str):
            continue
        term = _SPACE.sub(" ", value).strip().strip("\"'")
        if len(term) < 3 or len(term) > 60 or len(term.split()) > 5:
            continue
        if term.casefold() in seen:
            continue
        seen.add(term.casefold())
        terms.append(term)
    return terms[:MAX_TERMS]


class Expander:
    """Gets each interest's search terms, from the cache or from Claude."""

    def __init__(self, cli: ClaudeCLI, workers: int = 4) -> None:
        self.cli = cli
        self.workers = workers

    def cached(self, interest: Interest, cache: Cache) -> list[str] | None:
        return cache.get_expansion(cache_key(self.cli.model, interest))

    def expand(
        self,
        config: Config,
        cache: Cache,
        *,
        ask: bool = True,
        log: Callable[[str], None] = lambda message: None,
    ) -> tuple[Config, list[str]]:
        """The config with search terms added to its topic interests, plus notes.

        With `ask=False` (Claude not available), only cached terms are used.
        """
        topics = [i for i in config.interests if i.has_topic]
        terms: dict[str, list[str]] = {}
        missing: list[Interest] = []
        for interest in topics:
            found = self.cached(interest, cache)
            if found is None:
                missing.append(interest)
            else:
                terms[interest.name] = found

        notes: list[str] = []
        if missing and ask:
            count = len(missing)
            log(
                f"Asking Claude for search terms for {count}"
                f" interest{'s' if count != 1 else ''} (once per change)…"
            )

            def work(interest: Interest) -> tuple[Interest, list[str] | Exception]:
                try:
                    answer = self.cli.ask(SYSTEM_PROMPT, build_prompt(interest), SCHEMA)
                except ClaudeError as exc:
                    return interest, exc
                return interest, clean_terms(answer.data.get("terms"), interest)

            with ThreadPoolExecutor(max_workers=max(1, min(self.workers, count))) as pool:
                for interest, outcome in pool.map(work, missing):
                    if isinstance(outcome, Exception):
                        notes.append(
                            f"Claude couldn't suggest search terms for “{interest.name}”"
                            f" ({outcome}); it was matched by its own words only."
                        )
                        continue
                    cache.put_expansion(cache_key(self.cli.model, interest), outcome)
                    terms[interest.name] = outcome
        for name, found in terms.items():
            if found:
                log(f"  {name}: also searching for {', '.join(found)}")

        interests = [
            replace(i, search_terms=terms[i.name]) if terms.get(i.name) else i
            for i in config.interests
        ]
        return replace(config, interests=interests), notes


def with_cached_terms(config: Config, cache: Cache, model: str) -> Config:
    """The config with whatever search terms are cached (no Claude calls)."""
    interests = []
    for interest in config.interests:
        found = cache.get_expansion(cache_key(model, interest)) if interest.has_topic else None
        interests.append(replace(interest, search_terms=found) if found else interest)
    return replace(config, interests=interests)
