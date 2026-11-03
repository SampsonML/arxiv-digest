"""A digest as data.

Each run produces one Digest. It is saved in the database and rendered from
there as Markdown (`render.py`), as HTML (`html_render.py`) and in the web app,
so every view shows exactly the same papers, explanations and groups.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from . import __version__
from .config import Config
from .models import DeepRead, Paper
from .rank import Pick


@dataclass(frozen=True)
class Evidence:
    """The local match for the interest a paper is filed under."""

    similarity: float | None = None
    rank: int | None = None  # 1 = the day's closest paper to the interest
    of: int | None = None
    keywords: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()


@dataclass(frozen=True)
class Entry:
    paper: Paper
    interest: str
    why: str | None = None
    relevance: int | None = None
    followed_authors: tuple[str, ...] = ()
    evidence: Evidence | None = None
    deep_read: DeepRead | None = None


@dataclass
class Digest:
    generated: datetime  # local time
    categories: list[str]
    interests: list[str]  # in config order; groups follow it
    top_picks: int
    new_papers: int
    shortlisted: int
    window_start: datetime  # UTC
    window_end: datetime  # UTC
    ranked_by: str
    notes: list[str] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    version: str = __version__

    @property
    def title(self) -> str:
        g = self.generated
        return f"arXiv digest — {g:%a} {g.day} {g:%b %Y}"

    @property
    def top(self) -> list[Entry]:
        return self.entries[: self.top_picks]

    def groups(self) -> list[tuple[str, list[Entry]]]:
        """The entries after the top picks, by interest, in config order."""
        rest = self.entries[self.top_picks :]
        groups = [(name, [e for e in rest if e.interest == name]) for name in self.interests]
        known = set(self.interests)
        # Interests removed from the config since this digest was made.
        for entry in rest:
            if entry.interest not in known:
                known.add(entry.interest)
                groups.append((entry.interest, [e for e in rest if e.interest == entry.interest]))
        return [(name, items) for name, items in groups if items]

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generated": self.generated.isoformat(),
            "categories": self.categories,
            "interests": self.interests,
            "top_picks": self.top_picks,
            "new_papers": self.new_papers,
            "shortlisted": self.shortlisted,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "ranked_by": self.ranked_by,
            "notes": self.notes,
            "entries": [
                {
                    "paper": e.paper.to_dict(),
                    "interest": e.interest,
                    "why": e.why,
                    "relevance": e.relevance,
                    "followed_authors": list(e.followed_authors),
                    "evidence": None
                    if e.evidence is None
                    else {
                        "similarity": e.evidence.similarity,
                        "rank": e.evidence.rank,
                        "of": e.evidence.of,
                        "keywords": list(e.evidence.keywords),
                        "authors": list(e.evidence.authors),
                    },
                    "deep_read": None if e.deep_read is None else e.deep_read.to_dict(),
                }
                for e in self.entries
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Digest:
        entries = []
        for item in data.get("entries", []):
            ev = item.get("evidence")
            entries.append(
                Entry(
                    paper=Paper.from_dict(item["paper"]),
                    interest=item["interest"],
                    why=item.get("why"),
                    relevance=item.get("relevance"),
                    followed_authors=tuple(item.get("followed_authors", [])),
                    evidence=None
                    if ev is None
                    else Evidence(
                        similarity=ev.get("similarity"),
                        rank=ev.get("rank"),
                        of=ev.get("of"),
                        keywords=tuple(ev.get("keywords", [])),
                        authors=tuple(ev.get("authors", [])),
                    ),
                    deep_read=DeepRead.from_dict(item["deep_read"])
                    if item.get("deep_read")
                    else None,
                )
            )
        return cls(
            generated=datetime.fromisoformat(data["generated"]),
            categories=list(data["categories"]),
            interests=list(data["interests"]),
            top_picks=int(data["top_picks"]),
            new_papers=int(data["new_papers"]),
            shortlisted=int(data["shortlisted"]),
            window_start=datetime.fromisoformat(data["window_start"]),
            window_end=datetime.fromisoformat(data["window_end"]),
            ranked_by=data["ranked_by"],
            notes=list(data.get("notes", [])),
            entries=entries,
            version=data.get("version", ""),
        )


def build_digest(
    *,
    config: Config,
    picks: Sequence[Pick],
    new_papers: int,
    shortlisted: int,
    window_start: datetime,
    window_end: datetime,
    ranked_by: str,
    generated: datetime,
    notes: Sequence[str] = (),
) -> Digest:
    entries = []
    for pick in picks:
        candidate = pick.candidate
        match = candidate.match_for(pick.interest)  # evidence only for the filed interest
        evidence = None
        if match is not None:
            evidence = Evidence(
                similarity=match.similarity,
                rank=match.rank,
                of=match.of,
                keywords=match.keywords,
                authors=match.authors,
            )
        entries.append(
            Entry(
                paper=candidate.paper,
                interest=pick.interest,
                why=pick.why,
                relevance=pick.relevance,
                followed_authors=candidate.followed_authors,
                evidence=evidence,
                deep_read=pick.deep_read,
            )
        )
    return Digest(
        generated=generated,
        categories=list(config.categories),
        interests=[i.name for i in config.interests],
        top_picks=config.top_picks,
        new_papers=new_papers,
        shortlisted=shortlisted,
        window_start=window_start,
        window_end=window_end,
        ranked_by=ranked_by,
        notes=list(notes),
        entries=entries,
    )
