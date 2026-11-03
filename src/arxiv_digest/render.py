"""The digest as Markdown: top picks first, then the rest grouped by interest.

The output is plain CommonMark plus <details> blocks for abstracts, so it reads
well on GitHub, in Obsidian and in any Markdown viewer. LaTeX in titles and
abstracts ($...$) is left untouched so math renders where supported.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime

from .digest import Digest, Entry, Evidence
from .matching import match_authors
from .models import Paper

MAX_AUTHORS = 8
MAX_COMMENT = 120
_MATH = re.compile(r"(\$\$.+?\$\$|\$.+?\$)", re.S)
_TAG_START = re.compile(r"<(?=[A-Za-z/!?])")


def _safe(text: str) -> str:
    """Stop text like "<a few" being read as an HTML tag, without touching $math$."""
    parts = _MATH.split(text)
    return "".join(p if i % 2 else _TAG_START.sub("&lt;", p) for i, p in enumerate(parts))


def author_list(
    authors: Sequence[str], followed: Sequence[str]
) -> tuple[list[tuple[str, bool]], int]:
    """The authors to show, each with whether they are followed, and the total count.

    Long lists are cut to MAX_AUTHORS, but followed authors are always shown.
    """
    shown = [(name, bool(followed and match_authors([name], followed))) for name in authors]
    kept = shown[:MAX_AUTHORS] + [item for item in shown[MAX_AUTHORS:] if item[1]]
    return kept, len(authors)


def _authors(authors: Sequence[str], followed: Sequence[str]) -> str:
    kept, total = author_list(authors, followed)
    text = ", ".join(f"**{name}**" if is_followed else name for name, is_followed in kept)
    if total > MAX_AUTHORS:
        text += f" et al. ({total} authors)"
    return text


def evidence_parts(
    evidence: Evidence | None,
    interest: str,
    relevance: int | None,
    *,
    with_interest: bool,
    weird: bool = False,
) -> list[str]:
    """The pieces of the "Matched" line (shared with the HTML view)."""
    parts = []
    if with_interest:
        parts.append(interest)
    if relevance is not None:
        parts.append(f"{'weirdness' if weird else 'relevance'} {relevance}/10")
    if evidence is None:
        return parts
    if weird:  # "Find something weird": how isolated the paper is
        if evidence.rank is not None and evidence.of is not None:
            text = f"#{evidence.rank} most unusual of {evidence.of}"
            if evidence.similarity is not None:
                text += f" (similarity to its nearest neighbours {evidence.similarity:.2f})"
            parts.append(text)
        return parts
    if evidence.authors:
        parts.append("followed author: " + ", ".join(evidence.authors))
    if evidence.rank is not None and evidence.of is not None and evidence.similarity is not None:
        parts.append(
            f"#{evidence.rank} of {evidence.of} by meaning (similarity {evidence.similarity:.2f})"
        )
    elif evidence.similarity is not None:
        parts.append(f"similarity {evidence.similarity:.2f}")
    if evidence.keywords:
        parts.append("keywords: " + ", ".join(evidence.keywords))
    return parts


def short_comment(paper: Paper) -> str:
    comment = paper.comment
    if len(comment) > MAX_COMMENT:
        comment = comment[: MAX_COMMENT - 1].rstrip() + "…"
    return comment


def _links(paper: Paper) -> str:
    parts = [f"[arXiv:{paper.id}]({paper.abs_url})", f"[PDF]({paper.pdf_url})"]
    parts.append(paper.primary_category)
    if paper.comment:
        parts.append(_safe(short_comment(paper)))
    return " · ".join(parts)


def deep_read_source(entry: Entry) -> str:
    read = entry.deep_read
    assert read is not None
    return f"Claude read the full text ({read.source}{', cut short' if read.truncated else ''})"


def why_label(weird: bool) -> str:
    return "Why it's weird" if weird else "Why"


def deep_why_label(weird: bool) -> str:
    return "Why it's weird" if weird else "Why it matters"


def _deep_read(entry: Entry, weird: bool) -> list[str]:
    read = entry.deep_read
    assert read is not None
    return [
        f"**Deep read** · *{deep_read_source(entry)}*",
        "",
        f"- **Method:** {_safe(read.method)}",
        f"- **Results:** {_safe(read.results)}",
        f"- **{deep_why_label(weird)}:** {_safe(read.why)}",
        "",
    ]


def _entry(entry: Entry, heading: str, *, with_interest: bool, weird: bool = False) -> list[str]:
    paper = entry.paper
    lines = [
        f"{heading} {_safe(paper.title)}",
        "",
        _links(paper),
        "",
        _authors(paper.authors, entry.followed_authors),
        "",
    ]
    if entry.why:
        lines += [f"**{why_label(weird)}:** {_safe(entry.why)}", ""]
    if entry.deep_read:
        lines += _deep_read(entry, weird)
    parts = evidence_parts(
        entry.evidence, entry.interest, entry.relevance, with_interest=with_interest, weird=weird
    )
    if with_interest and parts:
        parts[0] = f"*{parts[0]}*"
    if parts:
        # With Claude's explanation present, the local evidence is a quieter footnote.
        word = "Found" if weird else "Matched"
        label = f"*{word}:*" if entry.why else f"**{word}:**"
        lines += [f"{label} " + " · ".join(parts), ""]
    lines += [
        "<details><summary>Abstract</summary>",
        "",
        _safe(paper.abstract),
        "",
        "</details>",
        "",
    ]
    return lines


def window_text(start: datetime, end: datetime) -> str:
    return f"{start:%d %b %H:%M} – {end:%d %b %H:%M} UTC"


def stats_text(digest: Digest) -> str:
    if digest.weird:
        return (
            f"{digest.new_papers} papers in the latest announcement in"
            f" {', '.join(digest.categories)} · {digest.shortlisted} most unusual"
            f" · {len(digest.entries)} picked"
        )
    return (
        f"{digest.new_papers} new papers in {', '.join(digest.categories)}"
        f" · {digest.shortlisted} shortlisted · {len(digest.entries)} picked"
    )


def footer_text(digest: Digest) -> str:
    return (
        f"Generated by arxiv-digest {digest.version} on {digest.generated:%Y-%m-%d %H:%M}"
        f" · papers submitted {window_text(digest.window_start, digest.window_end)}"
    )


def top_heading(digest: Digest) -> str:
    return "🌀 The weird ones" if digest.weird else "⭐ Top picks"


def empty_text(digest: Digest) -> str:
    if digest.weird:
        return "Nothing weird turned up this time."
    return "No new papers matched your interests this time."


def render_markdown(digest: Digest) -> str:
    lines = [f"# {digest.title}", "", stats_text(digest), "", f"> {digest.ranked_by}", ""]
    for note in digest.notes:
        lines += [f"> **Note:** {note}", ""]

    if not digest.entries:
        lines += [empty_text(digest), ""]
    else:
        if digest.top:
            lines += [f"## {top_heading(digest)}", ""]
            for number, entry in enumerate(digest.top, start=1):
                lines += _entry(
                    entry, f"### {number}.", with_interest=not digest.weird, weird=digest.weird
                )
        for interest, entries in digest.groups():
            lines += [f"## {interest} ({len(entries)})", ""]
            for entry in entries:
                lines += _entry(entry, "###", with_interest=False)

    lines += ["---", "", f"*{footer_text(digest)}*", ""]
    return "\n".join(lines)
