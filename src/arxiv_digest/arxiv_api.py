"""Fetching new papers from the arXiv API (https://info.arxiv.org/help/api/).

The API only lists papers once they have been announced, so asking for
everything *submitted* in a date range never returns papers early. A paper's
`published` time is when version 1 was submitted, so later versions of old
papers (replacements) fall outside the range automatically.

arXiv asks API users to wait at least three seconds between requests; the client
enforces that and retries temporary failures.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from datetime import time as clock_time
from zoneinfo import ZoneInfo

import httpx

from . import __version__
from .models import Paper

API_URL = "https://export.arxiv.org/api/query"
PAGE_SIZE = 500
MIN_INTERVAL = 3.0  # seconds between requests, as arXiv asks
MAX_ATTEMPTS = 4

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_ID_RE = re.compile(r"arxiv\.org/abs/(?P<id>.+?)(?:v(?P<version>\d+))?$")
_SPACE_RE = re.compile(r"\s+")


class ArxivError(Exception):
    """The arXiv API could not be reached or returned something unusable."""


def _clean(text: str | None) -> str:
    return _SPACE_RE.sub(" ", text or "").strip()


def _date(text: str | None) -> datetime:
    if not text:
        raise ArxivError("entry without a date")
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def parse_feed(xml_text: str) -> tuple[list[Paper], int]:
    """Papers in one API response, and the total number of results for the query."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ArxivError(f"arXiv returned invalid XML: {exc}") from exc
    total_text = root.findtext("opensearch:totalResults", namespaces=_NS)
    total = int(total_text) if total_text and total_text.strip().isdigit() else 0
    papers = []
    for entry in root.findall("atom:entry", _NS):
        raw_id = entry.findtext("atom:id", namespaces=_NS) or ""
        match = _ID_RE.search(raw_id.strip())
        if not match:
            # The API reports query errors as a single entry with an /api/errors id.
            message = _clean(entry.findtext("atom:summary", namespaces=_NS))
            raise ArxivError(f"arXiv rejected the query: {message or raw_id}")
        primary = entry.find("arxiv:primary_category", _NS)
        categories = tuple(
            c.get("term", "") for c in entry.findall("atom:category", _NS) if c.get("term")
        )
        papers.append(
            Paper(
                id=match.group("id"),
                version=int(match.group("version") or 1),
                title=_clean(entry.findtext("atom:title", namespaces=_NS)),
                abstract=_clean(entry.findtext("atom:summary", namespaces=_NS)),
                authors=tuple(
                    _clean(a.findtext("atom:name", namespaces=_NS))
                    for a in entry.findall("atom:author", _NS)
                ),
                categories=categories,
                primary_category=(primary.get("term") if primary is not None else "")
                or (categories[0] if categories else ""),
                published=_date(entry.findtext("atom:published", namespaces=_NS)),
                updated=_date(entry.findtext("atom:updated", namespaces=_NS)),
                comment=_clean(entry.findtext("arxiv:comment", namespaces=_NS)),
                journal_ref=_clean(entry.findtext("arxiv:journal_ref", namespaces=_NS)),
                doi=_clean(entry.findtext("arxiv:doi", namespaces=_NS)),
            )
        )
    return papers, total


ARXIV_TZ = ZoneInfo("America/New_York")
SUBMISSION_CUTOFF = clock_time(14, 0)  # weekdays, US Eastern


def announcement_start(newest: datetime) -> datetime:
    """Start of the submission window of the announcement that contains `newest`.

    Each arXiv announcement contains the submissions between two consecutive
    weekday cutoffs (14:00 US Eastern), so Monday's covers Friday 14:00 to
    Monday 14:00. Holidays shift the schedule; then the window is a day off,
    which only changes how much a first digest covers.
    """
    local = newest.astimezone(ARXIV_TZ)
    close = datetime.combine(local.date(), SUBMISSION_CUTOFF, ARXIV_TZ)
    if local > close:
        close += timedelta(days=1)
    while close.weekday() >= 5:  # Saturday, Sunday
        close += timedelta(days=1)
    opening = close - timedelta(days=1)
    while opening.weekday() >= 5:
        opening -= timedelta(days=1)
    return opening.astimezone(UTC)


def _api_time(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y%m%d%H%M")


def build_query(categories: Iterable[str], start: datetime, end: datetime) -> str:
    cats = " OR ".join(f"cat:{c}" for c in categories)
    return f"({cats}) AND submittedDate:[{_api_time(start)} TO {_api_time(end)}]"


class ArxivClient:
    def __init__(
        self,
        http: httpx.Client | None = None,
        *,
        min_interval: float = MIN_INTERVAL,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._http = http or httpx.Client(
            timeout=httpx.Timeout(60.0, connect=15.0),
            headers={"User-Agent": f"arxiv-digest/{__version__} (research paper digest)"},
            follow_redirects=True,
        )
        self._min_interval = min_interval
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None

    def close(self) -> None:
        self._http.close()

    def _wait_turn(self) -> None:
        if self._last_request is not None:
            remaining = self._min_interval - (self._clock() - self._last_request)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request = self._clock()

    def _get(self, params: dict[str, str | int]) -> str:
        delay = self._min_interval
        last_error = "unknown error"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._wait_turn()
            try:
                response = self._http.get(API_URL, params=params)
            except httpx.TransportError as exc:
                last_error = f"network error: {exc}"
            else:
                if response.status_code == 200:
                    return response.text
                last_error = f"HTTP {response.status_code}"
                if response.status_code not in (429, 500, 502, 503, 504):
                    break
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = max(delay, float(retry_after))
            if attempt < MAX_ATTEMPTS:
                self._sleep(delay)
                delay *= 2
        raise ArxivError(f"Could not fetch from the arXiv API ({last_error})")

    def search(
        self,
        categories: list[str],
        start: datetime,
        end: datetime,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[Paper]:
        """Every paper in `categories` whose first version was submitted in [start, end]."""
        query = build_query(categories, start, end)
        papers: dict[str, Paper] = {}
        offset = 0
        total = None
        empty_retries = 0
        while total is None or offset < total:
            text = self._get(
                {
                    "search_query": query,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "start": offset,
                    "max_results": PAGE_SIZE,
                }
            )
            page, page_total = parse_feed(text)
            total = page_total if total is None else total
            if not page:
                # The API occasionally returns an empty page mid-way; retrying helps.
                if offset < total and empty_retries < 3:
                    empty_retries += 1
                    continue
                break
            empty_retries = 0
            for paper in page:
                papers.setdefault(paper.id, paper)
            offset += len(page)
            if progress:
                progress(min(offset, total), total)
        return sorted(papers.values(), key=lambda p: p.published, reverse=True)

    def by_ids(self, ids: list[str]) -> list[Paper]:
        """Papers by arXiv id (latest metadata)."""
        if not ids:
            return []
        text = self._get({"id_list": ",".join(ids), "start": 0, "max_results": len(ids)})
        papers, _ = parse_feed(text)
        return papers

    def newest(self, categories: list[str]) -> Paper | None:
        """The most recently submitted (and announced) paper in `categories`."""
        query = "(" + " OR ".join(f"cat:{c}" for c in categories) + ")"
        text = self._get(
            {
                "search_query": query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "start": 0,
                "max_results": 1,
            }
        )
        papers, _ = parse_feed(text)
        return papers[0] if papers else None
