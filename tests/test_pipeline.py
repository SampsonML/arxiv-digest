from datetime import timedelta

from conftest import NOW, FakeClient, FakeEmbedder, make_paper

from arxiv_digest.pipeline import RunOptions, digest_paths, run_digest
from arxiv_digest.store import Store


def _papers():
    return [
        make_paper(
            "2611.00001", "Neural emulators speed up cosmological simulations", hours_ago=20
        ),
        make_paper("2611.00002", "An N-body emulator with transformers", hours_ago=21),
        make_paper("2611.00003", "Stellar populations", authors=("Vera Rubin",), hours_ago=22),
        make_paper("2611.00004", "Medieval poetry", hours_ago=23),
        make_paper("2611.00005", "Old news about emulators", hours_ago=24 * 6),
    ]


def test_first_run_writes_digest_and_remembers_papers(tmp_path, make_config):
    config = make_config()
    store = Store(tmp_path / "db.sqlite3")
    client = FakeClient(_papers())
    embedder = FakeEmbedder()

    result = run_digest(config, store, client, embedder, log=lambda m: None, now=NOW)

    # First run: one day of submissions before the newest paper; the 6-day-old one is out.
    assert {p.id for p in result.new_papers} == {
        "2611.00001",
        "2611.00002",
        "2611.00003",
        "2611.00004",
    }
    picked = {p.candidate.paper.id for p in result.picks}
    assert {"2611.00001", "2611.00002", "2611.00003"} <= picked
    assert "2611.00004" not in picked
    assert result.path is not None and result.path.exists()
    text = result.path.read_text()
    assert "## ⭐ Top picks" in text
    assert "[arXiv:2611.00003](https://arxiv.org/abs/2611.00003)" in text
    assert "**Vera Rubin**" in text  # followed author highlighted
    assert store.count_seen() == 4
    assert store.last_run()["picked"] == len(result.picks)


def test_second_run_shows_only_new_papers(tmp_path, make_config):
    config = make_config()
    store = Store(tmp_path / "db.sqlite3")
    papers = _papers()
    client = FakeClient(papers)
    run_digest(config, store, client, FakeEmbedder(), log=lambda m: None, now=NOW)

    later = NOW + timedelta(days=1)
    papers.append(make_paper("2611.00009", "A new emulator", hours_ago=-12))  # submitted later
    embedder = FakeEmbedder()
    result = run_digest(config, store, client, embedder, log=lambda m: None, now=later)

    assert [p.id for p in result.new_papers] == ["2611.00009"]
    start, end = client.calls[-1]
    assert start <= NOW - timedelta(hours=20)  # overlap margin before the newest seen paper
    assert end == later

    result = run_digest(config, store, client, embedder, log=lambda m: None, now=later)
    assert result.new_papers == [] and result.markdown is None


def test_dry_run_saves_nothing(tmp_path, make_config):
    config = make_config()
    store = Store(tmp_path / "db.sqlite3")
    options = RunOptions(dry_run=True)
    result = run_digest(
        config, store, FakeClient(_papers()), FakeEmbedder(), options, log=lambda m: None, now=NOW
    )
    assert result.markdown and result.path is None
    assert not config.output_dir.exists()
    assert store.count_seen() == 0 and store.last_run() is None


def test_include_seen_and_embedding_cache(tmp_path, make_config):
    config = make_config()
    store = Store(tmp_path / "db.sqlite3")
    client = FakeClient(_papers())
    run_digest(config, store, client, FakeEmbedder(), log=lambda m: None, now=NOW)
    embedder = FakeEmbedder()
    result = run_digest(
        config, store, client, embedder, RunOptions(include_seen=True), log=lambda m: None, now=NOW
    )
    assert len(result.new_papers) == 4
    assert embedder.passage_calls == 0  # vectors came from the cache
    assert result.path.name.endswith("-2.md")  # same day, so a second file


def test_without_model_ranks_by_keywords_and_authors(tmp_path, make_config):
    config = make_config()
    store = Store(tmp_path / "db.sqlite3")
    result = run_digest(config, store, FakeClient(_papers()), None, log=lambda m: None, now=NOW)
    assert {p.candidate.paper.id for p in result.picks} == {
        "2611.00001",
        "2611.00002",
        "2611.00003",
    }
    assert "keywords and followed authors only" in result.markdown


def test_digest_paths_number_repeat_days_and_keep_md_html_paired(tmp_path):
    md, html = digest_paths(tmp_path, NOW)
    assert (md.name, html.name) == ("2026-11-03.md", "2026-11-03.html")
    html.write_text("x")  # even a lone .html blocks the name
    md, html = digest_paths(tmp_path, NOW)
    assert (md.name, html.name) == ("2026-11-03-2.md", "2026-11-03-2.html")


def test_run_writes_html_and_saves_the_digest(tmp_path, make_config):
    config = make_config()
    store = Store(tmp_path / "db.sqlite3")
    result = run_digest(
        config, store, FakeClient(_papers()), FakeEmbedder(), log=lambda m: None, now=NOW
    )
    assert result.html_path.exists() and result.html_path.suffix == ".html"
    assert result.html_path.stem == result.path.stem
    html = result.html_path.read_text()
    assert html.startswith("<!doctype html>")
    assert 'href="https://arxiv.org/abs/2611.00003"' in html
    saved, row = store.get_digest(result.digest_id)
    assert row["html_path"] == str(result.html_path)
    assert [e.paper.id for e in saved.entries] == [p.candidate.paper.id for p in result.picks]
    assert store.list_digests()[0]["id"] == result.digest_id


class FakeRanker:
    label = "Claude (fake)"

    def __init__(self, choose=(), error=None, installed=True, signed_in=True):
        self.choose = list(choose)
        self.error = error
        self.installed = installed
        self._signed_in = signed_in
        self.seen_shortlist = None

    def available(self):
        return self.installed

    def signed_in(self):
        return self._signed_in

    def rank(self, config, shortlisted):
        from arxiv_digest.claude_cli import ClaudeAnswer, ClaudeError
        from arxiv_digest.rank import Pick, RankResult

        self.seen_shortlist = [c.paper.id for c in shortlisted]
        if isinstance(self.error, Exception):
            raise self.error
        if self.error:
            raise ClaudeError(self.error)
        by_id = {c.paper.id: c for c in shortlisted}
        picks = [
            Pick(by_id[pid], by_id[pid].best.interest, f"Because {pid} matters.", 9)
            for pid in self.choose
        ]
        return RankResult(picks, ClaudeAnswer(data={}, model="claude-test", duration_s=3.0))


def test_claude_chooses_and_explains(tmp_path, make_config):
    config = make_config()
    store = Store(tmp_path / "db.sqlite3")
    ranker = FakeRanker(choose=["2611.00002", "2611.00003"])
    result = run_digest(
        config,
        store,
        FakeClient(_papers()),
        FakeEmbedder(),
        ranker=ranker,
        log=lambda m: None,
        now=NOW,
    )
    assert "2611.00004" in ranker.seen_shortlist  # Claude sees the whole shortlist
    assert [p.candidate.paper.id for p in result.picks] == ["2611.00002", "2611.00003"]
    text = result.markdown
    assert "Chosen and explained by Claude (claude-test)" in text
    assert "**Why:** Because 2611.00002 matters." in text
    assert "relevance 9/10" in text
    assert "2611.00001" not in text  # Claude left it out, so the digest does too


def test_claude_failure_falls_back_to_local_picks(tmp_path, make_config):
    config = make_config()
    store = Store(tmp_path / "db.sqlite3")
    result = run_digest(
        config,
        store,
        FakeClient(_papers()),
        FakeEmbedder(),
        ranker=FakeRanker(error="usage limit reached"),
        log=lambda m: None,
        now=NOW,
    )
    assert {p.candidate.paper.id for p in result.picks} >= {"2611.00001", "2611.00003"}
    assert "Claude couldn't rank the papers (usage limit reached)" in result.markdown
    assert "Ranked on this computer" in result.markdown


def test_missing_claude_is_noted(tmp_path, make_config):
    config = make_config()
    store = Store(tmp_path / "db.sqlite3")
    result = run_digest(
        config,
        store,
        FakeClient(_papers()),
        FakeEmbedder(),
        ranker=FakeRanker(installed=False),
        log=lambda m: None,
        now=NOW,
    )
    assert "wasn't found" in result.markdown and result.picks


def test_no_claude_option_skips_ranker(tmp_path, make_config):
    config = make_config()
    store = Store(tmp_path / "db.sqlite3")
    ranker = FakeRanker(choose=["2611.00002"])
    run_digest(
        config,
        store,
        FakeClient(_papers()),
        FakeEmbedder(),
        RunOptions(use_claude=False),
        ranker=ranker,
        log=lambda m: None,
        now=NOW,
    )
    assert ranker.seen_shortlist is None


def test_signed_out_claude_is_skipped_with_a_sign_in_note(tmp_path, make_config):
    config = make_config()
    store = Store(tmp_path / "db.sqlite3")
    ranker = FakeRanker(choose=["2611.00002"], signed_in=False)
    result = run_digest(
        config,
        store,
        FakeClient(_papers()),
        FakeEmbedder(),
        ranker=ranker,
        log=lambda m: None,
        now=NOW,
    )
    assert ranker.seen_shortlist is None  # no call was attempted
    assert "isn't signed in" in result.markdown and "claude auth login" in result.markdown
    assert result.picks  # the local ranking is used instead


def test_sign_in_error_during_the_call_gets_the_same_note(tmp_path, make_config):
    from arxiv_digest.claude_cli import ClaudeNotSignedIn

    config = make_config()
    store = Store(tmp_path / "db.sqlite3")
    ranker = FakeRanker(error=ClaudeNotSignedIn("token expired"), signed_in=None)
    result = run_digest(
        config,
        store,
        FakeClient(_papers()),
        FakeEmbedder(),
        ranker=ranker,
        log=lambda m: None,
        now=NOW,
    )
    assert "isn't signed in" in result.markdown and "claude auth login" in result.markdown


class FakeDeepReader:
    label = "Claude (fake)"

    def __init__(self):
        self.asked = None

    def read(self, config, picks, cache=None, log=None):
        from arxiv_digest.models import DeepRead

        self.asked = [p.candidate.paper.id for p in picks]
        reads = {
            pid: DeepRead(f"Method of {pid}.", "Results.", "Matters.", "arXiv HTML")
            for pid in self.asked[:1]
        }
        return reads, ["Couldn't deep-read “x”: HTTP 404."]


def test_deep_read_of_top_picks(tmp_path, make_config):
    config = make_config(top_picks=2, deep_reads=3)
    store = Store(tmp_path / "db.sqlite3")
    reader = FakeDeepReader()
    result = run_digest(
        config,
        store,
        FakeClient(_papers()),
        FakeEmbedder(),
        ranker=FakeRanker(choose=["2611.00002", "2611.00003", "2611.00001"]),
        deep_reader=reader,
        log=lambda m: None,
        now=NOW,
    )
    assert reader.asked == ["2611.00002", "2611.00003"]  # only top picks (min of the two settings)
    assert result.picks[0].deep_read.method == "Method of 2611.00002."
    assert result.picks[1].deep_read is None
    assert "**Deep read** · *Claude read the full text (arXiv HTML)*" in result.markdown
    assert "- **Method:** Method of 2611.00002." in result.markdown
    assert "HTTP 404" in result.markdown  # the failure note
    assert 'class="deep-read"' in result.html
    saved, _ = store.get_digest(result.digest_id)
    assert saved.entries[0].deep_read.method == "Method of 2611.00002."


def test_no_deep_read_without_claude_ranking_or_when_off(tmp_path, make_config):
    config = make_config()
    reader = FakeDeepReader()
    run_digest(
        config,
        Store(tmp_path / "a.sqlite3"),
        FakeClient(_papers()),
        FakeEmbedder(),
        ranker=FakeRanker(error="down"),
        deep_reader=reader,
        log=lambda m: None,
        now=NOW,
    )
    assert reader.asked is None  # Claude couldn't rank, so no deep read either
    run_digest(
        config,
        Store(tmp_path / "b.sqlite3"),
        FakeClient(_papers()),
        FakeEmbedder(),
        RunOptions(deep_read=False),
        ranker=FakeRanker(choose=["2611.00002"]),
        deep_reader=reader,
        log=lambda m: None,
        now=NOW,
    )
    assert reader.asked is None


def test_store_caches_deep_reads(tmp_path):
    from arxiv_digest.models import DeepRead

    store = Store(tmp_path / "db.sqlite3")
    read = DeepRead("m", "r", "w", "PDF", True)
    assert store.get_deep_read("2611.00001", "k") is None
    store.put_deep_read("2611.00001", "k", read)
    assert store.get_deep_read("2611.00001", "k") == read
    assert store.get_deep_read("2611.00001", "other") is None
