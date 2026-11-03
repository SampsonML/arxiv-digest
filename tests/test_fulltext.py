import httpx
import pytest

from arxiv_digest import fulltext
from arxiv_digest.fulltext import FullTextError, FullTextFetcher, html_text

PAGE = """<html><head><title>x</title></head><body>
<nav class="ltx_page_navbar">Navigation should vanish</nav>
<article class="ltx_document">
<h1 class="ltx_title ltx_title_document">Neural <em>emulators</em></h1>
<div class="ltx_authors"><span class="ltx_creator">Ada Lovelace</span>
<span class="ltx_contact ltx_role_email">ada@example.org</span></div>
<div class="ltx_abstract"><h6 class="ltx_title">Abstract</h6>
<p class="ltx_p">We emulate <math alttext="N" display="inline"><semantics><mi>N</mi>
<annotation encoding="application/x-tex">N</annotation></semantics></math>-body runs.</p></div>
<section class="ltx_section"><h2 class="ltx_title ltx_title_section">1 Results</h2>
<div class="ltx_para"><p class="ltx_p">Error below <math alttext="1\\%"><mn>1</mn></math> at
<math alttext="z&lt;2"><mi>z</mi></math>.</p></div>
<table class="ltx_tabular"><tr><td>Model</td><td>Error</td></tr>
<tr><td>Ours</td><td>0.9%</td></tr></table>
<figure class="ltx_figure"><img src="x.png"/>
<figcaption>Figure 1: The emulator.</figcaption></figure>
</section>
<section class="ltx_bibliography"><h2>References</h2><ul><li>Some citation</li></ul></section>
</article>
<footer>Footer text</footer>
</body></html>"""


def test_html_text_keeps_structure_and_math():
    text = html_text(PAGE)
    assert text.startswith("## Neural emulators")
    assert "We emulate $N$ -body runs." in text
    assert "Error below $1\\%$ at $z<2$ ." in text
    assert "## 1 Results" in text
    assert "| Model | Error" in text and "| Ours | 0.9%" in text
    assert "Figure 1: The emulator." in text
    for gone in ("Navigation", "ada@example.org", "Some citation", "Footer", "x-tex"):
        assert gone not in text


def _fetcher(handler, sleeps=None):
    clock = {"t": 0.0}

    def sleep(seconds):
        (sleeps if sleeps is not None else []).append(seconds)
        clock["t"] += seconds

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return FullTextFetcher(http, sleep=sleep, clock=lambda: clock["t"])


LONG_PAGE = PAGE.replace("Figure 1: The emulator.", "Figure 1: " + "words " * 600)


def test_fetch_prefers_html():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, text=LONG_PAGE)

    text = _fetcher(handler).fetch("2611.00001")
    assert text.source == "arXiv HTML" and not text.truncated
    assert seen == ["https://arxiv.org/html/2611.00001"]


@pytest.mark.parametrize("html_status", [404, 429])
def test_fetch_falls_back_to_pdf(monkeypatch, html_status):
    monkeypatch.setattr(fulltext, "pdf_text", lambda data: "pdf words " * 400)
    sleeps = []

    def handler(request):
        if "/html/" in str(request.url):
            return httpx.Response(html_status, text="nope")
        assert str(request.url) == "https://export.arxiv.org/pdf/2611.00001"
        return httpx.Response(200, content=b"%PDF-1.4 ...")

    text = _fetcher(handler, sleeps).fetch("2611.00001")
    assert text.source == "PDF"
    assert sleeps == [fulltext.MIN_INTERVAL]  # waited between the two requests


def test_fetch_gives_up_clearly():
    with pytest.raises(FullTextError, match="HTTP 404"):
        _fetcher(lambda request: httpx.Response(404)).fetch("2611.00001")


def test_long_text_is_cut(monkeypatch):
    monkeypatch.setattr(fulltext, "MAX_CHARS", 3000)
    text, truncated = fulltext._cut("line\n" * 2000)
    assert truncated and len(text) <= 3000


def test_pdf_text_extracts_and_drops_references():
    content = (
        "BT /F1 12 Tf 72 720 Td (Introduction to emulators) Tj ET\n"
        "BT /F1 12 Tf 72 700 Td (We find errors below one percent.) Tj ET\n"
    )
    pdf = _minimal_pdf(content)
    text = fulltext.pdf_text(pdf)
    assert "Introduction to emulators" in text and "one percent" in text


def _minimal_pdf(stream: str) -> bytes:
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        " /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}endstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = "%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out.encode()))
        out += f"{number} 0 obj\n{body}\nendobj\n"
    xref = len(out.encode())
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return out.encode()
