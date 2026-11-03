"""Find crossovers: bridge maps, scoring, Claude's picks and the crossover digest."""

import json
import subprocess
import sys

from conftest import NOW, FakeClient, FakeEmbedder, make_paper

from arxiv_digest.claude_cli import ClaudeAnswer, ClaudeCLI
from arxiv_digest.config import Interest
from arxiv_digest.crossover import (
    Bridge,
    BridgeMap,
    Mapper,
    build_pick_prompt,
    map_schema,
    parse_map,
    parse_picks,
    run_crossovers,
    score_crossovers,
    search_categories,
    shortlist_crossovers,
)
from arxiv_digest.html_render import digest_body
from arxiv_digest.models import DeepRead
from arxiv_digest.pipeline import RunOptions
from arxiv_digest.rank import Pick
from arxiv_digest.store import Store

SBI = Interest(
    name="Simulation-based inference",
    description="neural inference of cosmological parameters from simulations",
)
MAP = {
    "problems": ["inferring the parameters of a noisy simulator without a likelihood"],
    "bridges": [
        {
            "category": "q-bio.PE",
            "field": "Population biology",
            "why": "Epidemic models are simulators fitted to noisy data.",
            "phrases": ["approximate bayesian computation", "x", "epidemic simulator"],
        },
        {"category": "astro-ph.CO", "field": "Cosmology", "why": "own field", "phrases": []},
        {"category": "not.a-category", "field": "?", "why": "?", "phrases": []},
        {"category": "q-bio.PE", "field": "dupe", "why": "dupe", "phrases": []},
    ],
}


class Runner:
    def __init__(self, data=MAP):
        self.calls = 0
        self.data = data

    def __call__(self, argv, **kwargs):
        self.calls += 1
        out = {"type": "result", "subtype": "success", "is_error": False}
        out["structured_output"] = self.data
        return subprocess.CompletedProcess(argv, 0, json.dumps(out), "")


def test_parse_map_keeps_valid_foreign_categories():
    bridge_map = parse_map(MAP, SBI, ["astro-ph.CO"])
    assert [b.category for b in bridge_map.bridges] == ["q-bio.PE"]  # own, bogus, dupe dropped
    assert bridge_map.bridges[0].phrases == (
        "approximate bayesian computation",
        "epidemic simulator",
    )
    assert bridge_map.problems[0].startswith("inferring the parameters")
    enum = map_schema(["astro-ph.CO"])["properties"]["bridges"]["items"]["properties"]["category"]
    assert "astro-ph.CO" not in enum["enum"] and "q-bio.PE" in enum["enum"]


def test_search_categories_take_turns_and_are_capped():
    def bridges(name, codes):
        return BridgeMap(name, (), tuple(Bridge(name, c, c, "", ()) for c in codes))

    maps = [bridges("a", ["x1", "x2", "x3"]), bridges("b", ["y1", "x1"])]
    assert search_categories(maps) == ["x1", "y1", "x2", "x3"]
    assert search_categories(maps, limit=2) == ["x1", "y1"]


def test_maps_are_cached(tmp_path, make_config):
    store = Store(tmp_path / "db.sqlite3")
    runner = Runner()
    mapper = Mapper(ClaudeCLI(sys.executable, "sonnet", runner=runner))
    config = make_config(interests=[SBI], categories=["astro-ph.CO"])
    maps, notes = mapper.maps(config, store)
    assert [b.category for b in maps[0].bridges] == ["q-bio.PE"] and notes == []
    again, _ = mapper.maps(config, store, ask=False)
    assert again == maps and runner.calls == 1


def _papers():
    own = [
        make_paper(
            f"2611.000{i:02d}",
            f"Cosmological parameter inference with neural networks {i}",
            "We infer cosmological parameters from galaxy surveys with neural networks.",
            categories=("astro-ph.CO",),
            hours_ago=20 + i / 10,
        )
        for i in range(10)
    ]
    foreign = [
        make_paper(
            "2611.00100",
            "Approximate Bayesian computation for an epidemic simulator",
            "Inferring the parameters of a noisy epidemic simulator without a likelihood.",
            categories=("q-bio.PE",),
            hours_ago=21,
        ),
        make_paper(
            "2611.00101",
            "Bird song syntax",
            "Songbirds sing in sequences.",
            categories=("q-bio.PE",),
            hours_ago=22,
        ),
    ]
    return own, foreign


def test_scoring_favours_the_same_problem_in_another_field():
    own, foreign = _papers()
    emb = FakeEmbedder()
    papers = own + foreign
    vectors = dict(
        zip([p.id for p in papers], emb.embed_passages([p.text for p in papers]), strict=True)
    )
    bridge_map = parse_map(MAP, SBI, ["astro-ph.CO"])
    problems = {SBI.name: emb.embed_queries(list(bridge_map.problems))}
    centroid = emb.embed_passages([own[0].text])[0]
    candidates = score_crossovers(foreign, [bridge_map], vectors, problems, centroid)
    best = shortlist_crossovers(candidates, [bridge_map])[0]
    assert best.paper.id == "2611.00100"
    assert best.best.keywords == ("approximate bayesian computation", "epidemic simulator")


class FakePicker:
    label = "Claude (fake)"

    def available(self):
        return True

    def signed_in(self):
        return True

    def pick(self, config, maps, candidates, picks):
        self.prompt = build_pick_prompt(config, maps, candidates, picks)
        chosen = [Pick(candidates[0], SBI.name, "Borrow their ABC sampler.", 8)]
        return chosen, ClaudeAnswer(data={}, model="claude-fake")


class FakeDeepReader:
    def read(self, config, picks, cache, log):
        read = DeepRead("They fit an epidemic.", "It works.", "Use their sampler.", "PDF", False)
        return {p.candidate.paper.id: read for p in picks}, []


def test_run_makes_a_crossover_digest(tmp_path, make_config):
    own, foreign = _papers()
    store = Store(tmp_path / "db.sqlite3")
    store.put_papers(own)
    config = make_config(interests=[SBI], categories=["astro-ph.CO"])
    picker = FakePicker()
    cross_listed = make_paper(
        "2611.00102", "Inference", "Simulator inference.", categories=("astro-ph.CO", "q-bio.PE")
    )
    result = run_crossovers(
        config,
        store,
        FakeClient([*foreign, cross_listed]),
        FakeEmbedder(),
        mapper=Mapper(ClaudeCLI(sys.executable, "sonnet", runner=Runner())),
        picker=picker,
        deep_reader=FakeDeepReader(),
        log=lambda m: None,
        now=NOW,
    )
    assert "Possible connection: Simulation-based inference, via q-bio.PE" in picker.prompt
    assert "2611.00102" not in {p.id for p in result.new_papers}  # its main category is yours
    digest = result.digest
    assert digest.kind == "crossover" and digest.title.startswith("Crossovers")
    assert digest.categories == ["q-bio.PE"]
    assert digest.bridges[0]["category"] == "q-bio.PE"
    md = result.markdown
    assert "## 🔭 Crossovers" in md and "**The crossover:** Borrow their ABC sampler." in md
    assert "- **What you could borrow:** Use their sampler." in md
    assert "transferability 8/10" in md and "Where Claude looked for crossovers" in md
    assert "Transferability 8/10" in digest_body(digest)
    saved, row = store.get_digest(result.digest_id)
    assert row["kind"] == "crossover" and saved.bridges == digest.bridges
    assert store.count_seen() == 0


def test_without_claude_or_a_cached_map_there_is_nothing_to_do(tmp_path, make_config):
    store = Store(tmp_path / "db.sqlite3")
    config = make_config(interests=[SBI], categories=["astro-ph.CO"])
    result = run_crossovers(
        config,
        store,
        FakeClient(_papers()[1]),
        FakeEmbedder(),
        RunOptions(mode="crossover", use_claude=False),
        mapper=Mapper(ClaudeCLI(sys.executable, "sonnet", runner=Runner())),
        picker=FakePicker(),
        log=lambda m: None,
        now=NOW,
    )
    assert result.digest is None and "needs Claude" in result.notes[-1]


def test_parse_picks_checks_ids_and_interests():
    own, foreign = _papers()
    bridge_map = parse_map(MAP, SBI, ["astro-ph.CO"])
    candidates = score_crossovers(foreign, [bridge_map], None, None, None)
    data = {
        "picks": [
            {"id": "2611.00101v1", "interest": "Unknown", "transferability": 0, "why": " Birds. "},
            {"id": "2611.00101", "interest": SBI.name, "transferability": 9, "why": "dupe"},
            {"id": "9999.99999", "interest": SBI.name, "transferability": 9, "why": "not offered"},
        ]
    }
    picks = parse_picks(data, candidates, [bridge_map], 5)
    assert [(p.candidate.paper.id, p.interest, p.relevance, p.why) for p in picks] == [
        ("2611.00101", SBI.name, 1, "Birds.")
    ]
