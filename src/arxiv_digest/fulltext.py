"""A paper's full text, for Claude's deep read.

arXiv's HTML version (https://arxiv.org/html/<id>, made with LaTeXML) is used
when it exists: it has the structure and keeps math as LaTeX. Otherwise the PDF
is downloaded (from export.arxiv.org, the mirror arXiv prefers programs to use)
and its text extracted with pypdf. References are left out, and very long
papers are cut to MAX_CHARS so a deep read stays quick and cheap.
"""

from __future__ import annotations

import io
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser

import httpx

from . import __version__

HTML_URL = "https://arxiv.org/html/{id}"
PDF_URL = "https://export.arxiv.org/pdf/{id}"
MAX_CHARS = 120_000  # about 30k tokens
MIN_CHARS = 2_000  # less than this means the HTML conversion failed
MIN_INTERVAL = 3.0  # seconds between requests to arXiv

_SPACE = re.compile(r"[ \t\r\f\v]+")
_ANY_SPACE = re.compile(r"\s+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


class FullTextError(Exception):
    """Neither the HTML nor the PDF text could be fetched."""


@dataclass(frozen=True)
class FullText:
    text: str
    source: str  # "arXiv HTML" or "PDF"
    truncated: bool


# ---------------------------------------------------------------------------
# HTML


_SKIP_CLASSES = (
    "ltx_bibliography",
    "ltx_TOC",
    "ltx_page_footer",
    "ltx_page_header",
    "ltx_contact",
)
_SKIP_TAGS = {"script", "style", "nav", "button", "svg", "annotation", "annotation-xml"}
_BLOCK_TAGS = {"p", "div", "section", "li", "tr", "figcaption", "table", "blockquote", "br"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_VOID_TAGS = {"br", "img", "hr", "input", "meta", "link", "col", "source", "wbr"}


class _ArticleText(HTMLParser):
    """Text of the <article> in an arXiv (LaTeXML) HTML page, math as $LaTeX$."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.in_article = 0
        self.skip = 0  # depth inside elements whose text is dropped
        self.math = 0  # depth inside <math>, replaced by its alttext

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID_TAGS:
            if tag == "br" and self.in_article and not (self.skip or self.math):
                self.parts.append("\n")
            return
        attr = dict(attrs)
        classes = attr.get("class") or ""
        if tag == "article":
            self.in_article += 1
        if not self.in_article:
            return
        if self.skip or self.math:
            if self.skip:
                self.skip += 1
            else:
                self.math += 1
            return
        if tag in _SKIP_TAGS or any(c in classes for c in _SKIP_CLASSES):
            self.skip = 1
            return
        if tag == "math":
            self.math = 1
            latex = (attr.get("alttext") or "").strip()
            if latex:
                self.parts.append(f" ${latex}$ ")
            return
        if tag in _HEADINGS:
            self.parts.append("\n\n## ")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS or not self.in_article:
            return
        if self.skip:
            self.skip -= 1
            return
        if self.math:
            self.math -= 1
            return
        if tag == "article":
            self.in_article -= 1
        elif tag in _HEADINGS or tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.in_article and not self.skip and not self.math:
            # As in a browser, line breaks in the source are just spaces;
            # only block elements start new lines.
            self.parts.append(_ANY_SPACE.sub(" ", data))

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [_SPACE.sub(" ", line).strip() for line in raw.split("\n")]
        return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def html_text(html: str) -> str:
    parser = _ArticleText()
    parser.feed(html)
    parser.close()
    return parser.text()


# ---------------------------------------------------------------------------
# PDF


_REFERENCES = re.compile(r"\n\s*(references|bibliography)\s*\n", re.I)


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    # Drop the reference list: the last "References" heading in the second half.
    matches = [m for m in _REFERENCES.finditer(text) if m.start() > len(text) // 2]
    if matches:
        text = text[: matches[-1].start()]
    lines = [_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


# ---------------------------------------------------------------------------
# Fetching


def _cut(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_CHARS:
        return text, False
    cut = text.rfind("\n", 0, MAX_CHARS)
    return text[: cut if cut > MAX_CHARS // 2 else MAX_CHARS], True


class FullTextFetcher:
    """Fetches papers' text, waiting MIN_INTERVAL seconds between requests."""

    def __init__(
        self,
        http: httpx.Client | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.http = http or httpx.Client(
            timeout=httpx.Timeout(60.0, connect=15.0),
            headers={"User-Agent": f"arxiv-digest/{__version__} (research paper digest)"},
            follow_redirects=True,
        )
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None

    def close(self) -> None:
        self.http.close()

    def _get(self, url: str) -> httpx.Response | None:
        if self._last is not None:
            wait = MIN_INTERVAL - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()
        try:
            return self.http.get(url)
        except httpx.HTTPError:
            return None

    def fetch(self, paper_id: str) -> FullText:
        response = self._get(HTML_URL.format(id=paper_id))
        if response is not None and response.status_code == 200 and "ltx_document" in response.text:
            text = html_text(response.text)
            if len(text) >= MIN_CHARS:
                text, truncated = _cut(text)
                return FullText(text, "arXiv HTML", truncated)
        response = self._get(PDF_URL.format(id=paper_id))
        if response is None or response.status_code != 200:
            status = "network error" if response is None else f"HTTP {response.status_code}"
            raise FullTextError(f"couldn't download the paper ({status})")
        try:
            text = pdf_text(response.content)
        except Exception as exc:  # pypdf raises many kinds of errors on odd PDFs
            raise FullTextError(f"couldn't read the PDF ({exc})") from exc
        if len(text) < MIN_CHARS:
            raise FullTextError("the PDF had no extractable text")
        text, truncated = _cut(text)
        return FullText(text, "PDF", truncated)
