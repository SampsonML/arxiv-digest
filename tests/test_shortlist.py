from conftest import FakeEmbedder, make_paper

from arxiv_digest.config import Interest
from arxiv_digest.shortlist import score_papers, select_locally, shortlist

SIMS = Interest(
    name="Fast simulations",
    description="neural emulators that speed up cosmological simulations",
    keywords=["emulator"],
    exclude=["quantum"],
)
PEOPLE = Interest(name="People", authors=["Vera Rubin"])


def _vectors(papers, interests):
    emb = FakeEmbedder()
    paper_vecs = dict(
        zip([p.id for p in papers], emb.embed_passages([p.text for p in papers]), strict=True)
    )
    topics = [i for i in interests if i.has_topic]
    query_vecs = dict(
        zip(
            [i.name for i in topics], emb.embed_queries([i.query_text for i in topics]), strict=True
        )
    )
    return paper_vecs, query_vecs


def test_exclude_require_keywords_and_authors():
    papers = [
        make_paper("1", "Neural emulators for cosmological simulations"),
        make_paper("2", "Quantum emulator for cosmological simulations"),
        make_paper("3", "Stellar spectra", authors=("V. Rubin",)),
        make_paper("4", "Unrelated topic"),
    ]
    required = Interest(name="Req", keywords=["simulations"], require=["neural", "cosmological"])
    candidates = {c.paper.id: c for c in score_papers(papers, [SIMS, PEOPLE, required])}
    assert candidates["1"].match_for("Fast simulations").keywords == ("emulator",)
    assert candidates["1"].match_for("Req") is not None
    assert "2" not in candidates  # excluded word; fails `require` too
    assert candidates["3"].followed_authors == ("Vera Rubin",)
    assert candidates["3"].score >= 1.0  # author bonus beats any topic score
    assert "4" not in candidates  # no keyword, author or similarity signal


def test_similarity_ranks_by_meaning_when_vectors_given():
    papers = [
        make_paper("close", "Neural emulators speed up cosmological simulations"),
        make_paper("far", "Medieval poetry and its meters"),
    ]
    candidates = score_papers(papers, [SIMS], *_vectors(papers, [SIMS]))
    by_id = {c.paper.id: c for c in candidates}
    assert by_id["close"].best.similarity > by_id["far"].best.similarity


def test_shortlist_keeps_authors_and_balances_topics():
    other = Interest(name="Lensing", description="weak gravitational lensing shear")
    papers = [
        make_paper(f"s{i}", f"Neural emulators cosmological simulations {i}") for i in range(10)
    ]
    papers += [make_paper("lens", "Weak gravitational lensing shear calibration")]
    papers += [make_paper("auth", "Something else", authors=("Vera Rubin",))]
    interests = [SIMS, other, PEOPLE]
    candidates = score_papers(papers, interests, *_vectors(papers, interests))
    chosen = shortlist(candidates, interests, size=6)
    ids = [c.paper.id for c in chosen]
    assert len(ids) == 6
    assert "auth" in ids and "lens" in ids


def test_select_locally_applies_threshold_but_keeps_keyword_and_author_hits():
    papers = [
        make_paper("kw", "An emulator for something"),
        make_paper("auth", "Anything", authors=("Vera Rubin",)),
        make_paper("weak", "Galaxies"),
    ]
    interests = [SIMS, PEOPLE]
    candidates = score_papers(papers, interests, *_vectors(papers, interests))
    picked = select_locally(candidates, max_papers=10, min_zscore=10)
    assert {c.paper.id for c in picked} == {"kw", "auth"}
    assert select_locally(candidates, max_papers=1, min_zscore=10)[0].paper.id == "auth"


def test_zscores_and_ranks_are_relative_to_the_day():
    papers = [make_paper("close", "Neural emulators speed up cosmological simulations")]
    papers += [make_paper(f"n{i}", f"Unrelated topic number {i} about poetry") for i in range(20)]
    candidates = {c.paper.id: c for c in score_papers(papers, [SIMS], *_vectors(papers, [SIMS]))}
    best = candidates["close"].best
    assert best.rank == 1 and best.of == 21
    assert best.zscore > 3
    picked = select_locally(list(candidates.values()), max_papers=10, min_zscore=2.5)
    assert [c.paper.id for c in picked] == ["close"]
