"""Keyword and author matching.

Text and terms are normalized the same way before matching: accents removed,
lower-cased, and every run of punctuation turned into a single space. So
"N-body" matches "N body", "n-body" and "N-Body"; "Gödel" matches "Godel". A
term matches whole words only, with an optional plural "s"/"es" ("emulator"
matches "emulators", but "lens" does not match "lensing").
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NON_ALNUM.sub(" ", stripped.lower()).strip()


@lru_cache(maxsize=4096)
def _term_pattern(term: str) -> re.Pattern[str] | None:
    norm = normalize(term)
    if not norm:
        return None
    return re.compile(r"(?<![a-z0-9])" + re.escape(norm) + r"(?:s|es)?(?![a-z0-9])")


def find_terms(normalized_text: str, terms: Iterable[str]) -> list[str]:
    """The terms (as written in the config) that occur in already-normalized text."""
    found = []
    for term in terms:
        pattern = _term_pattern(term)
        if pattern is not None and pattern.search(normalized_text):
            found.append(term)
    return found


def author_key(name: str) -> tuple[str, str] | None:
    """(surname, first initial) for a person's name.

    Accepts "First Middle Last" (arXiv's format) and "Last, First". The first
    initial is "" when only one name is given.
    """
    if "," in name:
        last, _, first = name.partition(",")
        last_tokens = normalize(last).split()
        first_tokens = normalize(first).split()
        if not last_tokens:
            return None
        surname = last_tokens[-1]
        initial = first_tokens[0][0] if first_tokens else ""
        return surname, initial
    tokens = [t for t in normalize(name).split() if t not in _NAME_SUFFIXES]
    if not tokens:
        return None
    if len(tokens) == 1:
        return tokens[0], ""
    return tokens[-1], tokens[0][0]


def match_authors(paper_authors: Iterable[str], followed: Iterable[str]) -> list[str]:
    """The followed names (as written in the config) found among a paper's authors.

    Names match on surname plus first initial, so "David W. Hogg", "D. W. Hogg"
    and "Hogg, David" all match each other. If either side has no first name,
    the surname alone decides.
    """
    paper_keys = [k for k in (author_key(a) for a in paper_authors) if k]
    found = []
    for name in followed:
        key = author_key(name)
        if key is None:
            continue
        surname, initial = key
        for p_surname, p_initial in paper_keys:
            if p_surname == surname and (not initial or not p_initial or p_initial == initial):
                found.append(name)
                break
    return found
