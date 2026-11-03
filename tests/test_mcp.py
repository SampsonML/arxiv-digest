import asyncio
import json
import sys
from datetime import UTC, datetime

import pytest
from conftest import FakeClient, FakeEmbedder, make_paper
from mcp.server.mcpserver.exceptions import ToolError

from arxiv_digest import config as cfg
from arxiv_digest import mcp_setup, paths
from arxiv_digest.digest import Digest, Entry
from arxiv_digest.fulltext import FullText
from arxiv_digest.mcp_server import clean_id, create_server
from arxiv_digest.pipeline import RunResult
from arxiv_digest.store import Store

PAPERS = [
    make_paper("2611.00001", "Neural emulators speed up cosmological simulations", hours_ago=20),
    make_paper("2611.00002", "Medieval poetry and its meters", hours_ago=21),
    make_paper("2611.00003", "Stellar populations", authors=("Vera Rubin",), hours_ago=22),
]


class Client(FakeClient):
    def by_ids(self, ids):
        return [make_paper(i, "Fetched from arXiv") for i in ids]

    def close(self):
        pass


class Fetcher:
    def fetch(self, paper_id):
        return FullText("0123456789" * 10, "arXiv HTML", False)

    def close(self):
        pass


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    monkeypatch.setenv("ARXIV_DIGEST_HOME", str(tmp_path / "home"))
    path = tmp_path / "config.toml"
    interests = [
        cfg.Interest(
            name="Fast sims",
            description="neural emulators cosmological simulations",
            keywords=["emulator"],
        ),
        cfg.Interest(name="People", authors=["Vera Rubin"]),
    ]
    path.write_text(cfg.render_template(["astro-ph.CO"], str(tmp_path / "digests"), interests))
    return path


def _server(config_path, run=None):
    return create_server(
        config_path,
        run=run or (lambda c, o, log: RunResult(datetime.now(UTC), datetime.now(UTC))),
        client_factory=lambda: Client(PAPERS),
        fetcher_factory=Fetcher,
        embedder_factory=lambda conf: FakeEmbedder(),
    )


def call(server, tool, **arguments):
    result = asyncio.run(server.call_tool(tool, arguments))
    return result.content[0].text


def test_tools_and_their_hints(config_path):
    tools = {t.name: t for t in asyncio.run(_server(config_path).list_tools())}
    assert set(tools) == {
        "list_digests",
        "get_digest",
        "search_papers",
        "get_paper",
        "todays_candidates",
        "run_digest",
        "list_interests",
        "add_interest",
        "update_interest",
        "remove_interest",
    }
    assert tools["list_digests"].annotations.read_only_hint is True
    assert tools["remove_interest"].annotations.destructive_hint is True
    assert tools["run_digest"].annotations.read_only_hint is False
    assert "untrusted" in _server(config_path).instructions


def test_digests(config_path):
    server = _server(config_path)
    assert "No digests yet" in call(server, "list_digests")
    digest = Digest(
        generated=datetime(2026, 11, 3, 9, 30).astimezone(),
        categories=["astro-ph.CO"],
        interests=["Fast sims"],
        top_picks=1,
        new_papers=616,
        shortlisted=35,
        window_start=datetime(2026, 11, 1, tzinfo=UTC),
        window_end=datetime(2026, 11, 2, tzinfo=UTC),
        ranked_by="Chosen by Claude.",
        entries=[Entry(PAPERS[0], "Fast sims", why="Speeds up sims.", relevance=9)],
    )
    store = Store(paths.database_path())
    digest_id = store.save_digest(digest, None, None)
    store.close()
    listing = call(server, "list_digests")
    assert f"Digest #{digest_id}" in listing and "1 picked of 616" in listing
    assert "top: Neural emulators" in listing
    text = call(server, "get_digest")
    assert text.startswith(f"Digest #{digest_id}") and "**Why:** Speeds up sims." in text
    with pytest.raises(ToolError, match="no digest #99"):
        call(server, "get_digest", digest_id=99)


def _seed(config_path):
    store = Store(paths.database_path())
    store.mark_seen(PAPERS, "x")
    store.close()


def test_search_papers_by_meaning(config_path):
    server = _server(config_path)
    assert "No stored papers" in call(server, "search_papers", query="emulators")
    _seed(config_path)
    text = call(
        server,
        "search_papers",
        query="neural emulators for cosmological simulations",
        days=30,
        limit=2,
    )
    assert text.index("Neural emulators") < text.index("###", text.index("Neural emulators") + 3)
    assert "Similarity:" in text
    with pytest.raises(ToolError, match="search query"):
        call(server, "search_papers", query="  ")


def test_get_paper_and_full_text_in_parts(config_path):
    server = _server(config_path)
    _seed(config_path)
    stored = call(server, "get_paper", arxiv_id="https://arxiv.org/abs/2611.00001v2")
    assert "Neural emulators" in stored and "Abstract:" in stored
    fetched = call(server, "get_paper", arxiv_id="arXiv:2611.09999")
    assert "Fetched from arXiv" in fetched
    part = call(
        server, "get_paper", arxiv_id="2611.00001", full_text=True, start=10, max_chars=1000
    )
    assert "<paper_text>\n0123456789" in part and "characters 10–100 of 100" in part
    assert "untrusted data" in part


def test_clean_id():
    assert clean_id("2611.02627v3") == "2611.02627"
    assert clean_id("https://arxiv.org/pdf/2611.02627v1.pdf") == "2611.02627"
    assert clean_id("arXiv:astro-ph/0601001") == "astro-ph/0601001"
    with pytest.raises(ToolError, match="doesn't look like"):
        clean_id("not an id")


def test_todays_candidates_marks_nothing_seen(config_path):
    text = call(_server(config_path), "todays_candidates", limit=5)
    assert "latest announcement has 3 papers" in text
    assert "Neural emulators" in text and "FOLLOWED AUTHOR: Vera Rubin" in text
    store = Store(paths.database_path())
    assert store.count_seen() == 0
    store.close()


def test_run_digest(config_path):
    digest = Digest(
        generated=datetime(2026, 11, 3, 9, 30).astimezone(),
        categories=["astro-ph.CO"],
        interests=["Fast sims"],
        top_picks=1,
        new_papers=10,
        shortlisted=5,
        ranked_by="r",
        window_start=datetime(2026, 11, 1, tzinfo=UTC),
        window_end=datetime(2026, 11, 2, tzinfo=UTC),
        entries=[Entry(PAPERS[0], "Fast sims", why="w")],
    )
    seen = {}

    def run(conf, options, log):
        seen["options"] = options
        result = RunResult(datetime.now(UTC), datetime.now(UTC), digest=digest, digest_id=7)
        result.notes = ["A note."]
        return result

    text = call(_server(config_path, run), "run_digest", include_seen=True, deep_read=False)
    assert text.startswith("Digest #7") and "Note: A note." in text and "Neural emulators" in text
    assert seen["options"].include_seen is True and seen["options"].deep_read is False
    assert "No new papers" in call(_server(config_path), "run_digest")


def test_interests(config_path):
    server = _server(config_path)
    assert "- Fast sims" in call(server, "list_interests")
    text = call(server, "add_interest", name="Lensing", keywords=["shear"])
    assert "- Lensing" in text and "keywords: shear" in text
    text = call(
        server, "update_interest", name="lensing", description="weak lensing", new_name="WL"
    )
    assert "- WL" in text and "description: weak lensing" in text and "keywords: shear" in text
    assert "- WL" not in call(server, "remove_interest", name="WL")
    with pytest.raises(ToolError, match="already an interest"):
        call(server, "add_interest", name="fast sims", keywords=["x"])
    with pytest.raises(ToolError, match="No interest called"):
        call(server, "remove_interest", name="Nope")
    with pytest.raises(ToolError, match="needs a `description`"):
        call(server, "add_interest", name="Empty")
    assert "# arxiv-digest configuration" in config_path.read_text()  # comments kept


def test_claude_code_registration_command(monkeypatch):
    monkeypatch.setenv("ARXIV_DIGEST_HOME", "/data/ad")
    argv = mcp_setup.claude_code_add("/bin/claude")
    assert argv[:6] == ["/bin/claude", "mcp", "add", "--scope", "user", "arxiv-digest"]
    assert argv[6:9] == ["-e", "ARXIV_DIGEST_HOME=/data/ad", "--"]
    assert argv[9:] == [sys.executable, "-m", "arxiv_digest", "mcp", "serve"]


def test_claude_desktop_config_is_edited_safely(tmp_path):
    path = tmp_path / "claude_desktop_config.json"
    path.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}, "theme": "dark"}))
    mcp_setup.add_to_desktop(path)
    data = json.loads(path.read_text())
    assert data["theme"] == "dark" and "other" in data["mcpServers"]
    entry = data["mcpServers"]["arxiv-digest"]
    assert entry["command"] == sys.executable and entry["args"][-2:] == ["mcp", "serve"]
    assert list(tmp_path.glob("claude_desktop_config.json.bak.*"))  # backed up first
    assert mcp_setup.remove_from_desktop(path)
    assert "arxiv-digest" not in json.loads(path.read_text())["mcpServers"]
    assert not mcp_setup.remove_from_desktop(path)
