from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from arxiv_digest.arxiv_api import (
    ArxivClient,
    ArxivError,
    announcement_start,
    build_query,
    parse_feed,
)

FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_page.xml"


def test_parse_feed_reads_entries():
    papers, total = parse_feed(FIXTURE.read_text())
    assert total == 3
    first = papers[0]
    assert first.id == "2611.00001"
    assert first.version == 2
    assert first.title == "Neural emulators for $N$-body simulations"
    assert first.abstract == "We train a neural emulator that replaces expensive N-body runs."
    assert first.authors == ("Ada Lovelace", "José Núñez")
    assert first.categories == ("astro-ph.CO", "cs.LG")
    assert first.primary_category == "astro-ph.CO"
    assert first.published == datetime(2026, 11, 1, 10, tzinfo=UTC)
    assert (first.comment, first.journal_ref, first.doi) == (
        "12 pages, 4 figures",
        "ApJ 1 (2026)",
        "10.0000/example",
    )
    assert first.abs_url == "https://arxiv.org/abs/2611.00001"
    assert papers[1].id == "astro-ph/0601001"
    assert papers[2].version == 1
    assert papers[2].primary_category == "stat.ML"


def test_parse_feed_reports_api_errors():
    xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/api/errors#incorrect_id_format</id>
      <summary>incorrect id format</summary></entry></feed>"""
    with pytest.raises(ArxivError, match="incorrect id format"):
        parse_feed(xml)


def test_build_query():
    start = datetime(2026, 11, 1, 9, 5, tzinfo=UTC)
    end = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)
    assert build_query(["astro-ph.CO", "cs.LG"], start, end) == (
        "(cat:astro-ph.CO OR cat:cs.LG) AND submittedDate:[202611010905 TO 202611031500]"
    )


def _feed(ids: list[str], total: int) -> str:
    entries = "".join(
        f"""<entry><id>http://arxiv.org/abs/{pid}v1</id><title>T {pid}</title>
        <summary>S</summary><published>2026-11-02T10:00:00Z</published>
        <updated>2026-11-02T10:00:00Z</updated>
        <category term="cs.LG"/><author><name>A B</name></author></entry>"""
        for pid in ids
    )
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
        f"<opensearch:totalResults>{total}</opensearch:totalResults>{entries}</feed>"
    )


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _client(handler, fake_time):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return ArxivClient(http, sleep=fake_time.sleep, clock=fake_time.clock)


def test_search_pages_through_results_and_waits_between_requests(monkeypatch):
    monkeypatch.setattr("arxiv_digest.arxiv_api.PAGE_SIZE", 2)
    pages = {0: ["2611.00001", "2611.00002"], 2: ["2611.00003"]}
    seen_starts = []

    def handler(request):
        start = int(request.url.params["start"])
        seen_starts.append(start)
        assert request.url.params["sortBy"] == "submittedDate"
        return httpx.Response(200, text=_feed(pages[start], total=3))

    fake_time = FakeTime()
    client = _client(handler, fake_time)
    now = datetime(2026, 11, 3, tzinfo=UTC)
    papers = client.search(["cs.LG"], now, now)
    assert [p.id for p in papers] == ["2611.00001", "2611.00002", "2611.00003"]
    assert seen_starts == [0, 2]
    assert fake_time.sleeps == [3.0]  # one wait between the two requests


def test_search_retries_temporary_errors():
    responses = [httpx.Response(503), httpx.Response(200, text=_feed(["2611.00001"], total=1))]

    def handler(request):
        return responses.pop(0)

    fake_time = FakeTime()
    now = datetime(2026, 11, 3, tzinfo=UTC)
    papers = _client(handler, fake_time).search(["cs.LG"], now, now)
    assert [p.id for p in papers] == ["2611.00001"]
    assert fake_time.sleeps  # backed off before retrying


def test_search_gives_up_on_permanent_errors():
    fake_time = FakeTime()
    client = _client(lambda request: httpx.Response(400), fake_time)
    now = datetime(2026, 11, 3, tzinfo=UTC)
    with pytest.raises(ArxivError, match="HTTP 400"):
        client.search(["cs.LG"], now, now)


@pytest.mark.parametrize(
    ("newest", "expected_start"),
    [
        # Monday 13:59 EST: Monday's announcement covers Friday 14:00 EDT onwards.
        (datetime(2026, 11, 2, 18, 59, tzinfo=UTC), datetime(2026, 10, 30, 18, 0, tzinfo=UTC)),
        # Wednesday 10:00 EST: Tuesday 14:00 to Wednesday 14:00.
        (datetime(2026, 11, 4, 15, 0, tzinfo=UTC), datetime(2026, 11, 3, 19, 0, tzinfo=UTC)),
        # Saturday submission: held for Monday's announcement (Friday 14:00 onwards).
        (datetime(2026, 11, 7, 15, 0, tzinfo=UTC), datetime(2026, 11, 6, 19, 0, tzinfo=UTC)),
        # Just after Tuesday's cutoff: belongs to Wednesday's announcement.
        (datetime(2026, 11, 3, 19, 30, tzinfo=UTC), datetime(2026, 11, 3, 19, 0, tzinfo=UTC)),
    ],
)
def test_announcement_start(newest, expected_start):
    assert announcement_start(newest) == expected_start


def test_newest_asks_for_one_paper():
    def handler(request):
        assert request.url.params["max_results"] == "1"
        assert "submittedDate" not in request.url.params["search_query"]
        return httpx.Response(200, text=_feed(["2611.00042"], total=99))

    paper = _client(handler, FakeTime()).newest(["cs.LG"])
    assert paper is not None and paper.id == "2611.00042"
