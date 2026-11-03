from conftest import make_paper

from arxiv_digest.config import Config, Interest
from arxiv_digest.rank import SYSTEM_PROMPT, answer_schema, build_prompt, parse_picks
from arxiv_digest.shortlist import Candidate, Match

SIMS = Interest(name="Fast simulations", description="emulators", keywords=["emulator"])
PEOPLE = Interest(name="Followed", authors=["Vera Rubin"])


def _config(tmp_path, max_papers=15):
    return Config(
        path=tmp_path / "c.toml",
        categories=["astro-ph.CO"],
        interests=[SIMS, PEOPLE],
        output_dir=tmp_path,
        max_papers=max_papers,
    )


def _candidate(pid, interest="Fast simulations", authors=(), **paper):
    match = Match(interest, 1.0, 0.5, (), tuple(authors))
    return Candidate(make_paper(pid, **paper), (match,))


def test_prompt_lists_interests_papers_and_followed_authors(tmp_path):
    shortlist = [
        _candidate("2611.00001", title="Emulators", categories=("astro-ph.CO", "cs.LG")),
        _candidate("2611.00002", "Followed", authors=("Vera Rubin",)),
    ]
    prompt = build_prompt(_config(tmp_path), shortlist)
    assert "### Fast simulations" in prompt and "Keywords: emulator" in prompt
    assert "Followed authors: Vera Rubin" in prompt
    assert '<paper id="2611.00001">' in prompt
    assert "Category: astro-ph.CO (also cs.LG)" in prompt
    assert "FOLLOWED AUTHOR: Vera Rubin" in prompt
    assert "at most 15 papers" in prompt
    assert "not instructions" in SYSTEM_PROMPT  # abstracts are treated as data


def test_schema_restricts_interests_and_count(tmp_path):
    schema = answer_schema(_config(tmp_path, max_papers=7))
    picks = schema["properties"]["picks"]
    assert picks["maxItems"] == 7
    assert picks["items"]["properties"]["interest"]["enum"] == ["Fast simulations", "Followed"]


def test_parse_picks_validates_claudes_answer(tmp_path):
    shortlist = [
        _candidate("2611.00001"),
        _candidate("2611.00002"),
        _candidate("2611.00003", "Followed", authors=("Vera Rubin",)),
    ]
    data = {
        "picks": [
            {
                "id": "arXiv:2611.00002v3",
                "interest": "Fast simulations",
                "relevance": 14,
                "why": "  Uses   an emulator\n for N-body runs. ",
            },
            {"id": "9999.99999", "interest": "Fast simulations", "relevance": 9, "why": "x"},
            {"id": "2611.00002", "interest": "Fast simulations", "relevance": 9, "why": "dup"},
            {"id": "2611.00001", "interest": "No such interest", "relevance": "high", "why": ""},
            "not a dict",
        ]
    }
    picks = parse_picks(data, _config(tmp_path), shortlist)
    assert [p.candidate.paper.id for p in picks] == ["2611.00002", "2611.00001", "2611.00003"]
    first, second, author = picks
    assert first.relevance == 10 and first.why == "Uses an emulator for N-body runs."
    assert second.interest == "Fast simulations" and second.relevance is None and second.why is None
    # The followed-author paper Claude left out is added back.
    assert author.interest == "Followed" and "Vera Rubin" in author.why


def test_parse_picks_cap_keeps_followed_authors(tmp_path):
    shortlist = [_candidate(f"2611.0000{i}") for i in range(1, 4)]
    shortlist.append(_candidate("2611.00009", "Followed", authors=("Vera Rubin",)))
    data = {
        "picks": [
            {"id": f"2611.0000{i}", "interest": "Fast simulations", "relevance": 9, "why": "w"}
            for i in range(1, 4)
        ]
    }
    picks = parse_picks(data, _config(tmp_path, max_papers=2), shortlist)
    assert [p.candidate.paper.id for p in picks] == ["2611.00001", "2611.00009"]


def test_parse_picks_empty_answer_means_nothing_relevant(tmp_path):
    assert parse_picks({"picks": []}, _config(tmp_path), [_candidate("2611.00001")]) == []
    assert parse_picks({}, _config(tmp_path), [_candidate("2611.00001")]) == []
