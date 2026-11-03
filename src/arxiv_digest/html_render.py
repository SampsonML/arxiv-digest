"""The digest as HTML: the same content and layout as the Markdown, in a page.

`digest_body` is the shared part (used by the standalone files and the web
app); `standalone_page` wraps it into a self-contained .html file with the
stylesheet inlined. All text from arXiv and from Claude is HTML-escaped. Math
($...$) is rendered in the browser by KaTeX, loaded from a CDN with pinned
versions and Subresource Integrity hashes; offline, it shows as plain text.
"""

from __future__ import annotations

from functools import cache
from html import escape
from importlib import resources

from .digest import Digest, Entry
from .render import MAX_AUTHORS, author_list, evidence_parts, footer_text, short_comment, stats_text

KATEX_VERSION = "0.19.1"
_KATEX = f"https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist"
KATEX_HEAD = (
    f'<link rel="stylesheet" href="{_KATEX}/katex.min.css" crossorigin="anonymous"'
    ' integrity="sha384-WwGtaC6ywCHxA5cstpj/3+xIZvTez8dq2IcBQaXRkenHZ7FVVq1ZROtURwfbtTlG">'
)
KATEX_SCRIPTS = (
    f'<script defer src="{_KATEX}/katex.min.js" crossorigin="anonymous"'
    ' integrity="sha384-a1WQdjRj162i159OYq041X6I4K8NenkrI09LlPmiHg0YOb7CR+wJOraXwK9abpnE">'
    "</script>\n"
    f'<script defer src="{_KATEX}/contrib/auto-render.min.js" crossorigin="anonymous"'
    ' integrity="sha384-bjyGPfbij8/NDKJhSGZNP/khQVgtHUE5exjm4Ydllo42FwIgYsdLO2lXGmRBf5Mz">'
    "</script>\n"
    "<script>\n"
    'document.addEventListener("DOMContentLoaded", function () {\n'
    '  if (typeof renderMathInElement !== "function") return;  // offline: leave $...$ as text\n'
    "  renderMathInElement(document.body, {\n"
    "    delimiters: [\n"
    '      {left: "$$", right: "$$", display: true},\n'
    '      {left: "$", right: "$", display: false}\n'
    "    ],\n"
    "    throwOnError: false\n"
    "  });\n"
    "});\n"
    "</script>"
)


@cache
def stylesheet() -> str:
    return resources.files("arxiv_digest").joinpath("assets/digest.css").read_text("utf-8")


def _e(text: str) -> str:
    return escape(text, quote=True)


def _link(url: str, text: str, cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return f'<a{attr} href="{_e(url)}" target="_blank" rel="noopener">{_e(text)}</a>'


def _authors(entry: Entry) -> str:
    kept, total = author_list(entry.paper.authors, entry.followed_authors)
    names = [
        f'<span class="followed">{_e(name)}</span>' if followed else _e(name)
        for name, followed in kept
    ]
    text = ", ".join(names)
    if total > MAX_AUTHORS:
        text += f" et al. ({total} authors)"
    return text


def _entry(entry: Entry, number: int | None) -> str:
    paper = entry.paper
    num = f'<span class="num">{number}.</span> ' if number is not None else ""
    meta = [
        _link(paper.abs_url, f"arXiv:{paper.id}"),
        _link(paper.pdf_url, "PDF"),
        _e(paper.primary_category),
    ]
    if paper.comment:
        meta.append(_e(short_comment(paper)))
    parts = [
        f'<div class="paper{" top" if number is not None else ""}" id="paper-{_e(paper.id)}">',
        f"<h3>{num}{_link(paper.abs_url, paper.title)}</h3>",
        f'<p class="meta">{" · ".join(meta)}</p>',
        f'<p class="authors">{_authors(entry)}</p>',
    ]
    if entry.why:
        parts.append(f'<p class="why"><strong>Why:</strong> {_e(entry.why)}</p>')
    evidence = evidence_parts(
        entry.evidence, entry.interest, entry.relevance, with_interest=number is not None
    )
    if evidence:
        if number is not None:
            evidence[0] = f'<span class="tag">{_e(evidence[0])}</span>'
            rest = [_e(p) for p in evidence[1:]]
            line = " · ".join([evidence[0], *rest]) if rest else evidence[0]
        else:
            line = " · ".join(_e(p) for p in evidence)
        parts.append(f'<p class="matched">Matched: {line}</p>')
    parts += [
        "<details><summary>Abstract</summary>",
        f"<p>{_e(paper.abstract)}</p>",
        "</details>",
        "</div>",
    ]
    return "\n".join(parts)


def digest_body(digest: Digest) -> str:
    """The digest as an <article>: header, top picks, interest groups, footer."""
    out = [
        '<article class="digest">',
        "<header>",
        f"<h1>{_e(digest.title)}</h1>",
        f'<p class="stats">{_e(stats_text(digest))}</p>',
        f'<p class="ranked-by">{_e(digest.ranked_by)}</p>',
    ]
    for note in digest.notes:
        out.append(f'<p class="note"><strong>Note:</strong> {_e(note)}</p>')
    out.append("</header>")
    if not digest.entries:
        out.append('<p class="empty">No new papers matched your interests this time.</p>')
    else:
        if digest.top:
            out += ['<section class="top-picks">', "<h2>⭐ Top picks</h2>"]
            out += [_entry(entry, number) for number, entry in enumerate(digest.top, start=1)]
            out.append("</section>")
        for interest, entries in digest.groups():
            out += [
                '<section class="interest">',
                f'<h2>{_e(interest)} <span class="count">({len(entries)})</span></h2>',
            ]
            out += [_entry(entry, None) for entry in entries]
            out.append("</section>")
    out += [f"<footer>{_e(footer_text(digest))}</footer>", "</article>"]
    return "\n".join(out)


def standalone_page(digest: Digest) -> str:
    """A complete, self-contained HTML file for one digest."""
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_e(digest.title)}</title>",
            KATEX_HEAD,
            f"<style>\n{stylesheet()}</style>",
            "</head>",
            "<body>",
            digest_body(digest),
            KATEX_SCRIPTS,
            "</body>",
            "</html>",
            "",
        ]
    )
