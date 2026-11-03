"""Find something weird: isolation scoring, Claude's picks, and the weird digest."""

from conftest import NOW, FakeClient, FakeEmbedder, make_paper

from arxiv_digest.claude_cli import ClaudeAnswer, ClaudeError
from arxiv_digest.html_render import digest_body
from arxiv_digest.models import DeepRead
from arxiv_digest.pipeline import RunOptions
from arxiv_digest.rank import Pick
from arxiv_digest.store import Store
from arxiv_digest.weird import (
    INTEREST,
    PICKS,
    answer_schema,
    build_prompt,
    parse_picks,
    run_weird,
    unusual_papers,
)


def _papers():
    usual = [
        make_paper(
            f"2611.000{i:02d}",
            f"Neural emulators for cosmological simulations {i}",
            "We train neural emulators of cosmological simulations of large scale structure.",
            hours_ago=20 + i / 10,
        )
        for i in range(12)
    ]
    odd = [
        make_paper(
            "2611.00100",
            "Bayesian inference of medieval bell ringing",
            "Church bells, monks and Markov chains.",
            hours_ago=21,
        ),
        make_paper(
            "2611.00101",
            "Galaxy shapes from sheep",
            "Pastoral herding dynamics as a model of tidal alignments.",
            categories=("astro-ph.CO", "q-bio.PE"),
            hours_ago=22,
        ),
        make_paper(
            "2611.00102",
            "Erratum: emulators",
            "Corrects a typo.",
            hours_ago=23,
        ),
    ]
    return usual + odd


def _vectors(papers):
    emb = FakeEmbedder()
    return dict(
        zip([p.id for p in papers], emb.embed_passages([p.text for p in papers]), strict=True)
    )


def test_the_odd_papers_come_first_and_errata_are_skipped():
    papers = _papers()
    oddities = unusual_papers(papers, _vectors(papers), size=4)
    top = [o.paper.id for o in oddities[:2]]
    assert set(top) == {"2611.00100", "2611.00101"}
    assert "2611.00102" not in {o.paper.id for o in unusual_papers(papers, _vectors(papers), 20)}
    assert oddities[0].rank == 1 and oddities[0].of == len(papers) - 1
    # Without vectors, a rare mix of fields still stands out.
    assert unusual_papers(papers, None, size=1)[0].paper.id == "2611.00101"


def test_prompt_schema_and_parsing():
    papers = _papers()
    oddities = unusual_papers(papers, _vectors(papers), size=5)
    prompt = build_prompt(["astro-ph.CO"], oddities, 5)
    assert '<paper id="2611.00100">' in prompt and "Choose the 5 weirdest" in prompt
    assert answer_schema(5)["properties"]["picks"]["maxItems"] == 5
    data = {
        "picks": [
            {"id": "arXiv:2611.00101v2", "weirdness": 12, "why": "  Sheep   as tidal tracers. "},
            {"id": "2611.00101", "weirdness": 5, "why": "duplicate"},
            {"id": "9999.99999", "weirdness": 9, "why": "not a candidate"},
            {"id": "2611.00100", "weirdness": 8, "why": "Bells."},
        ]
    }
    picks = parse_picks(data, oddities, 5)
    assert [p.candidate.paper.id for p in picks] == ["2611.00101", "2611.00100"]
    assert picks[0].relevance == 10 and picks[0].why == "Sheep as tidal tracers."
    assert picks[0].interest == INTEREST


class FakePicker:
    label = "Claude (fake)"

    def __init__(self, error=None, signed_in=True):
        self.error = error
        self._signed_in = signed_in

    def available(self):
        return True

    def signed_in(self):
        return self._signed_in

    def pick(self, categories, oddities, picks):
        if self.error:
            raise ClaudeError(self.error)
        chosen = [
            Pick(o.candidate, INTEREST, f"Weird because {o.paper.id}.", 9) for o in oddities[:2]
        ]
        return chosen, ClaudeAnswer(data={}, model="claude-fake")


class FakeDeepReader:
    def read(self, config, picks, cache, log):
        reads = {
            p.candidate.paper.id: DeepRead("Did a thing.", "Found a thing.", "Bells?", "PDF", False)
            for p in picks
        }
        return reads, []


def test_run_weird_makes_a_weird_digest_and_marks_nothing_seen(tmp_path, make_config):
    store = Store(tmp_path / "db.sqlite3")
    result = run_weird(
        make_config(),
        store,
        FakeClient(_papers()),
        FakeEmbedder(),
        picker=FakePicker(),
        deep_reader=FakeDeepReader(),
        log=lambda m: None,
        now=NOW,
    )
    assert len(result.picks) == 2 and result.picks[0].deep_read is not None
    digest = result.digest
    assert digest.weird and digest.title.startswith("Something weird")
    assert digest.groups() == []  # every entry is in the one weird section
    md = result.markdown
    assert "## 🌀 The weird ones" in md and "**Why it's weird:** Weird because" in md
    assert "- **Why it's weird:** Bells?" in md and "weirdness 9/10" in md
    assert "most unusual of" in md
    html = digest_body(digest)
    assert 'class="digest weird"' in html and "Weirdness 9/10" in html
    saved, row = store.get_digest(result.digest_id)
    assert saved.weird and row["kind"] == "weird"
    assert store.count_seen() == 0  # normal digests are unaffected
    assert store.get_paper("2611.00100") is not None  # but papers are kept for search


def test_without_claude_the_most_isolated_papers_are_shown(tmp_path, make_config):
    store = Store(tmp_path / "db.sqlite3")
    result = run_weird(
        make_config(),
        store,
        FakeClient(_papers()),
        FakeEmbedder(),
        RunOptions(mode="weird", use_claude=False),
        picker=FakePicker(),
        log=lambda m: None,
        now=NOW,
    )
    assert len(result.picks) == PICKS and all(p.why is None for p in result.picks)
    assert "Claude wasn't used" in result.digest.ranked_by

    failed = run_weird(
        make_config(),
        store,
        FakeClient(_papers()),
        FakeEmbedder(),
        picker=FakePicker(error="overloaded"),
        deep_reader=FakeDeepReader(),
        log=lambda m: None,
        now=NOW,
    )
    assert "Claude couldn't pick (overloaded)" in failed.notes[0]
    assert all(p.deep_read is None for p in failed.picks)  # no reads without Claude's picks
