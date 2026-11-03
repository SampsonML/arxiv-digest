"""Claude's search terms for interests, and keyword hits in the shortlist."""

import json
import subprocess
import sys

from conftest import NOW, FakeClient, FakeEmbedder, make_paper

from arxiv_digest.claude_cli import ClaudeCLI
from arxiv_digest.config import Interest
from arxiv_digest.expand import Expander, build_prompt, cache_key, clean_terms, with_cached_terms
from arxiv_digest.pipeline import run_digest
from arxiv_digest.shortlist import score_papers, shortlist
from arxiv_digest.store import Store

FLOWS = Interest(name="Flow matching", description="Papers about flow matching")


class Runner:
    def __init__(self, terms=("flow matching", "rectified flow"), fail=False):
        self.calls = 0
        self.terms = list(terms)
        self.fail = fail

    def __call__(self, argv, **kwargs):
        self.calls += 1
        if self.fail:
            out = {"type": "result", "subtype": "success", "is_error": True, "result": "overloaded"}
        else:
            out = {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "structured_output": {"terms": self.terms},
            }
        return subprocess.CompletedProcess(argv, 0, json.dumps(out), "")


def _expander(runner):
    return Expander(ClaudeCLI(sys.executable, "sonnet", runner=runner))


def test_prompt_and_cleaning():
    prompt = build_prompt(Interest(name="Lensing", description="weak lensing", keywords=["shear"]))
    assert "Interest: Lensing" in prompt and "own keywords: shear" in prompt
    interest = Interest(name="x", keywords=["Shear"])
    raw = [
        "  rectified   flow ",
        "shear",
        "RECTIFIED FLOW",
        "ab",
        3,
        "a b c d e f g",
        "flow matching",
    ]
    assert clean_terms(raw, interest) == ["rectified flow", "flow matching"]


def test_terms_are_cached_and_used_without_claude(tmp_path, make_config):
    store = Store(tmp_path / "db.sqlite3")
    runner = Runner()
    config = make_config(interests=[FLOWS])
    expanded, notes = _expander(runner).expand(config, store)
    assert expanded.interests[0].search_terms == ["flow matching", "rectified flow"]
    assert notes == [] and runner.calls == 1
    _expander(runner).expand(config, store)
    assert runner.calls == 1  # cached
    # Not asked (Claude unavailable): cached terms still apply.
    cached, _ = _expander(Runner(fail=True)).expand(config, store, ask=False)
    assert cached.interests[0].search_terms == ["flow matching", "rectified flow"]
    assert with_cached_terms(config, store, "sonnet").interests[0].search_terms
    # A changed interest asks again.
    assert cache_key("sonnet", FLOWS) != cache_key("sonnet", Interest(name="Flow matching"))


def test_a_failed_call_leaves_a_note(tmp_path, make_config):
    store = Store(tmp_path / "db.sqlite3")
    expanded, notes = _expander(Runner(fail=True)).expand(make_config(interests=[FLOWS]), store)
    assert expanded.interests[0].search_terms == []
    assert "couldn't suggest search terms" in notes[0]


def test_keyword_hits_get_reserved_places_in_the_shortlist():
    # Many papers that look closer "by meaning" than the ones with the phrase.
    noise = [make_paper(f"n{i}", f"Matching flows in graphs part {i}") for i in range(30)]
    hits = [
        make_paper(f"h{i}", f"A generative study {i}", "We use rectified flow.") for i in range(3)
    ]
    papers = noise + hits
    emb = FakeEmbedder()
    vectors = dict(
        zip([p.id for p in papers], emb.embed_passages([p.text for p in papers]), strict=True)
    )
    interest = Interest(name="Flows", description="matching flows", search_terms=["rectified flow"])
    queries = {"Flows": emb.embed_queries([interest.query_text])[0]}
    candidates = score_papers(papers, [interest], vectors, queries)
    short = shortlist(candidates, [interest], 6)
    assert {"h0", "h1", "h2"} <= {c.paper.id for c in short}
    assert len(short) == 6


def test_pipeline_uses_claudes_terms(tmp_path, make_config):
    papers = [
        make_paper("2611.00001", "Optical flow estimation", hours_ago=20),
        make_paper("2611.00002", "Generative models", "We train a rectified flow.", hours_ago=21),
    ]
    store = Store(tmp_path / "db.sqlite3")

    class Ranker:
        label = "Claude (fake)"

        def available(self):
            return True

        def signed_in(self):
            return True

        def rank(self, config, shortlisted):
            from arxiv_digest.rank import RankResult

            self.terms = config.interests[0].search_terms
            return RankResult(picks=[])

    ranker = Ranker()
    result = run_digest(
        make_config(interests=[FLOWS]),
        store,
        FakeClient(papers),
        FakeEmbedder(),
        ranker=ranker,
        expander=_expander(Runner()),
        log=lambda m: None,
        now=NOW,
    )
    assert ranker.terms == ["flow matching", "rectified flow"]
    match = next(c for c in result.shortlisted if c.paper.id == "2611.00002").best
    assert match.keywords == ("rectified flow",)
