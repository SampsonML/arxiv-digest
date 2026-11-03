"""The daily Markdown digest: top picks first, then the rest grouped by interest.

The output is plain CommonMark plus <details> blocks for abstracts, so it reads
well on GitHub, in Obsidian and in any Markdown viewer. LaTeX in titles and
abstracts ($...$) is left untouched so math renders where supported.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from . import __version__
from .config import Config
from .matching import match_authors
from .rank import Pick
from .shortlist import Candidate, Match

MAX_AUTHORS = 8
MAX_COMMENT = 120
_MATH = re.compile(r"(\$\$.+?\$\$|\$.+?\$)", re.S)
_TAG_START = re.compile(r"<(?=[A-Za-z/!?])")


@dataclass
class DigestStats:
    new_papers: int
    shortlisted: int
    window_start: datetime
    window_end: datetime
    ranked_by: str  # one sentence on how papers were ranked


def _safe(text: str) -> str:
    """Stop text like "<a few" being read as an HTML tag, without touching $math$."""
    parts = _MATH.split(text)
    return "".join(p if i % 2 else _TAG_START.sub("&lt;", p) for i, p in enumerate(parts))


def _authors(candidate: Candidate) -> str:
    authors = candidate.paper.authors
    followed = candidate.followed_authors
    shown = []
    for name in authors[:MAX_AUTHORS]:
        shown.append(f"**{name}**" if followed and match_authors([name], followed) else name)
    # Always show followed authors, even if they are far down a long author list.
    for name in authors[MAX_AUTHORS:]:
        if followed and match_authors([name], followed):
            shown.append(f"**{name}**")
    text = ", ".join(shown)
    if len(authors) > MAX_AUTHORS:
        text += f" et al. ({len(authors)} authors)"
    return text


def _match_line(
    match: Match | None, interest: str, relevance: int | None, *, with_interest: bool, label: str
) -> str:
    parts = []
    if with_interest:
        parts.append(f"*{interest}*")
    if relevance is not None:
        parts.append(f"relevance {relevance}/10")
    if match is None:
        return f"{label} " + " · ".join(parts) if parts else ""
    if match.authors:
        parts.append("followed author: " + ", ".join(match.authors))
    if match.rank is not None and match.of is not None:
        parts.append(f"#{match.rank} of {match.of} by meaning (similarity {match.similarity:.2f})")
    elif match.similarity is not None:
        parts.append(f"similarity {match.similarity:.2f}")
    if match.keywords:
        parts.append("keywords: " + ", ".join(match.keywords))
    return f"{label} " + " · ".join(parts)


def _links(candidate: Candidate) -> str:
    paper = candidate.paper
    parts = [
        f"[arXiv:{paper.id}]({paper.abs_url})",
        f"[PDF]({paper.pdf_url})",
        paper.primary_category,
    ]
    if paper.comment:
        comment = paper.comment
        if len(comment) > MAX_COMMENT:
            comment = comment[: MAX_COMMENT - 1].rstrip() + "…"
        parts.append(_safe(comment))
    return " · ".join(parts)


def _entry(pick: Pick, heading: str, *, with_interest: bool) -> list[str]:
    candidate = pick.candidate
    paper = candidate.paper
    # Local evidence only for the interest the paper is filed under.
    match = candidate.match_for(pick.interest)
    lines = [f"{heading} {_safe(paper.title)}", "", _links(candidate), "", _authors(candidate), ""]
    if pick.why:
        lines += [f"**Why:** {_safe(pick.why)}", ""]
    # With Claude's explanation present, the local evidence is a quieter footnote.
    label = "*Matched:*" if pick.why else "**Matched:**"
    matched = _match_line(
        match, pick.interest, pick.relevance, with_interest=with_interest, label=label
    )
    if matched:
        lines += [matched, ""]
    lines += [
        "<details><summary>Abstract</summary>",
        "",
        _safe(paper.abstract),
        "",
        "</details>",
        "",
    ]
    return lines


def _window(start: datetime, end: datetime) -> str:
    return f"{start:%d %b %H:%M} – {end:%d %b %H:%M} UTC"


def render_digest(
    *,
    config: Config,
    picks: Sequence[Pick],
    stats: DigestStats,
    generated: datetime,
    notes: Sequence[str] = (),
) -> str:
    lines = [
        f"# arXiv digest — {generated:%a} {generated.day} {generated:%b %Y}",
        "",
        f"{stats.new_papers} new papers in {', '.join(config.categories)}"
        f" · {stats.shortlisted} shortlisted · {len(picks)} picked",
        "",
        f"> {stats.ranked_by}",
        "",
    ]
    for note in notes:
        lines += [f"> **Note:** {note}", ""]

    if not picks:
        lines += ["No new papers matched your interests this time.", ""]
    else:
        top = list(picks[: config.top_picks])
        rest = list(picks[config.top_picks :])
        if top:
            lines += ["## ⭐ Top picks", ""]
            for number, pick in enumerate(top, start=1):
                lines += _entry(pick, f"### {number}.", with_interest=True)
        for interest in config.interests:
            group = [p for p in rest if p.interest == interest.name]
            if not group:
                continue
            lines += [f"## {interest.name} ({len(group)})", ""]
            for pick in group:
                lines += _entry(pick, "###", with_interest=False)

    lines += [
        "---",
        "",
        f"*Generated by arxiv-digest {__version__} on {generated:%Y-%m-%d %H:%M}"
        f" · papers submitted {_window(stats.window_start, stats.window_end)}*",
        "",
    ]
    return "\n".join(lines)
