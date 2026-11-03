import json
import subprocess
import sys

from conftest import make_paper

from arxiv_digest.claude_cli import ClaudeCLI
from arxiv_digest.config import Config, Interest
from arxiv_digest.deepread import DeepReader, build_prompt, cache_key, parse_read
from arxiv_digest.fulltext import FullText, FullTextError
from arxiv_digest.models import DeepRead
from arxiv_digest.rank import Pick
from arxiv_digest.shortlist import Candidate, Match

SIMS = Interest(name="Fast sims", description="emulators for N-body", keywords=["emulator"])
TEXT = FullText("## 1 Results\nErrors below 1%.", "arXiv HTML", False)


def _config(tmp_path):
    return Config(
        path=tmp_path / "c.toml", categories=["astro-ph.CO"], interests=[SIMS], output_dir=tmp_path
    )


def _pick(pid, title="A paper"):
    candidate = Candidate(make_paper(pid, title=title), (Match("Fast sims", 1.0, 0.5, (), ()),))
    return Pick(candidate, "Fast sims", why="w", relevance=9)


class Runner:
    def __init__(self, fail_for=()):
        self.prompts = []
        self.fail_for = fail_for

    def __call__(self, argv, **kwargs):
        self.prompts.append(kwargs["input"])
        if any(f in kwargs["input"] for f in self.fail_for):
            out = {"type": "result", "subtype": "success", "is_error": True, "result": "overloaded"}
        else:
            out = {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "structured_output": {
                    "method": "Trains an emulator.",
                    "results": "Errors below 1%.",
                    "why_it_matters": "Speeds up your sims.",
                },
            }
        return subprocess.CompletedProcess(argv, 0, json.dumps(out), "")


class DictCache:
    def __init__(self):
        self.data = {}

    def get_deep_read(self, paper_id, key):
        return self.data.get((paper_id, key))

    def put_deep_read(self, paper_id, key, read):
        self.data[(paper_id, key)] = read


def test_prompt_contains_interest_and_full_text():
    prompt = build_prompt(_pick("2611.00001", "Neural emulators"), SIMS, TEXT)
    assert "Name: Fast sims" in prompt and "Description: emulators for N-body" in prompt
    assert "Title: Neural emulators" in prompt
    assert "<paper_text>\n## 1 Results\nErrors below 1%.\n</paper_text>" in prompt
    cut = build_prompt(_pick("2611.00001"), SIMS, FullText("x", "PDF", True))
    assert "from the PDF; cut short" in cut


def test_cache_key_depends_on_model_and_interest():
    base = cache_key("sonnet", SIMS, "Fast sims")
    assert base == cache_key("sonnet", SIMS, "Fast sims")
    assert base != cache_key("opus", SIMS, "Fast sims")
    changed = Interest(name="Fast sims", description="something else")
    assert base != cache_key("sonnet", changed, "Fast sims")


def test_parse_read_cleans_and_validates():
    read = parse_read(
        {"method": "  Trains\n an emulator. ", "results": "Good.", "why_it_matters": "x" * 900},
        TEXT,
    )
    assert read.method == "Trains an emulator." and read.source == "arXiv HTML"
    assert len(read.why) <= 600 and read.why.endswith("…")
    sentences = "One short sentence. " * 50
    cut = parse_read({"method": sentences, "results": "r", "why_it_matters": "w"}, TEXT)
    assert cut.method.endswith("sentence.") and len(cut.method) <= 600  # whole sentences


def test_deep_reader_reads_in_parallel_caches_and_reports_failures(tmp_path):
    runner = Runner(fail_for=["Title: Broken"])
    fetched = []

    def fetch(paper_id):
        fetched.append(paper_id)
        if paper_id == "2611.00003":
            raise FullTextError("couldn't download the paper (HTTP 404)")
        return TEXT

    reader = DeepReader(ClaudeCLI(sys.executable, "sonnet", runner=runner), fetch)
    picks = [
        _pick("2611.00001", "Good"),
        _pick("2611.00002", "Broken"),
        _pick("2611.00003", "Gone"),
    ]
    cache = DictCache()
    reads, notes = reader.read(_config(tmp_path), picks, cache)
    assert list(reads) == ["2611.00001"]
    assert reads["2611.00001"] == DeepRead(
        "Trains an emulator.", "Errors below 1%.", "Speeds up your sims.", "arXiv HTML", False
    )
    assert any("Broken" in n and "overloaded" in n for n in notes)
    assert any("Gone" in n and "HTTP 404" in n for n in notes)
    assert fetched == ["2611.00001", "2611.00002", "2611.00003"]

    # A second run uses the cache: no download, no Claude call for the cached paper.
    fetched.clear()
    calls_before = len(runner.prompts)
    reads, _ = reader.read(_config(tmp_path), picks[:1], cache)
    assert "2611.00001" in reads and fetched == [] and len(runner.prompts) == calls_before
