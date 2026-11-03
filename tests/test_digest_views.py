from datetime import UTC, datetime

from conftest import make_paper

from arxiv_digest.digest import Digest, Entry, Evidence
from arxiv_digest.html_render import KATEX_VERSION, digest_body, standalone_page
from arxiv_digest.render import render_markdown


def _digest(**overrides):
    entries = [
        Entry(
            make_paper(
                "2611.00001",
                title="Emulators for $N$-body <b>sims</b> & more",
                abstract='Abstract with "quotes" and $\\sigma_8 < 1$.',
                authors=("Ada Lovelace", "Vera Rubin"),
            ),
            "Fast sims",
            why="Builds an emulator <script>alert(1)</script>.",
            relevance=9,
            followed_authors=("Vera Rubin",),
            evidence=Evidence(0.53, 1, 616, ("emulator",), ("Vera Rubin",)),
        ),
        Entry(make_paper("2611.00002", title="Second"), "Fast sims"),
        Entry(make_paper("2611.00003", title="Third"), "Removed interest"),
    ]
    values = dict(
        generated=datetime(2026, 11, 3, 9, 30).astimezone(),
        categories=["astro-ph.CO", "cs.LG"],
        interests=["Fast sims", "Lensing"],
        top_picks=1,
        new_papers=616,
        shortlisted=35,
        window_start=datetime(2026, 10, 30, 19, tzinfo=UTC),
        window_end=datetime(2026, 11, 2, 19, tzinfo=UTC),
        ranked_by="Chosen and explained by Claude (test).",
        notes=["A note"],
        entries=entries,
    )
    values.update(overrides)
    return Digest(**values)


def test_round_trip_through_dict():
    digest = _digest()
    again = Digest.from_dict(digest.to_dict())
    assert again == digest


def test_groups_follow_config_order_and_keep_removed_interests():
    groups = _digest().groups()
    assert [(name, [e.paper.id for e in items]) for name, items in groups] == [
        ("Fast sims", ["2611.00002"]),
        ("Removed interest", ["2611.00003"]),
    ]


def test_markdown_view():
    text = render_markdown(_digest())
    assert text.startswith("# arXiv digest — Tue 3 Nov 2026\n")
    assert "616 new papers in astro-ph.CO, cs.LG · 35 shortlisted · 3 picked" in text
    assert "### 1. Emulators for $N$-body &lt;b>sims&lt;/b> & more" in text
    assert "**Why:** Builds an emulator &lt;script>" in text
    assert "*Matched:* *Fast sims* · relevance 9/10 · followed author: Vera Rubin" in text
    assert "## Fast sims (1)" in text and "## Removed interest (1)" in text
    assert "> **Note:** A note" in text


def test_html_view_escapes_and_links():
    body = digest_body(_digest())
    assert "<script>alert" not in body  # Claude's text is escaped
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert "&lt;b&gt;sims&lt;/b&gt; &amp; more" in body
    assert "$N$-body" in body and "$\\sigma_8 &lt; 1$" in body  # math left for KaTeX
    assert (
        '<a href="https://arxiv.org/abs/2611.00001" target="_blank" rel="noopener">'
        "Emulators for" in body
    )
    assert 'href="https://arxiv.org/pdf/2611.00001"' in body
    assert '<span class="followed">Vera Rubin</span>' in body
    assert '<span class="tag">Fast sims</span>' in body
    assert "#1 of 616 by meaning (similarity 0.53)" in body
    assert body.count('class="paper top"') == 1 and body.count('class="paper"') == 2


def test_standalone_page_is_complete():
    page = standalone_page(_digest())
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    assert "<title>arXiv digest — Tue 3 Nov 2026</title>" in page
    assert f"katex@{KATEX_VERSION}" in page and 'integrity="sha384-' in page
    assert "--accent" in page  # stylesheet inlined


def test_empty_digest():
    digest = _digest(entries=[])
    assert "No new papers matched" in render_markdown(digest)
    assert "No new papers matched" in digest_body(digest)
