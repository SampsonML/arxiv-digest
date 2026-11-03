import threading
from datetime import UTC, datetime

import pytest
from conftest import make_paper
from starlette.testclient import TestClient

from arxiv_digest import config as cfg
from arxiv_digest import paths
from arxiv_digest.digest import Digest, Entry
from arxiv_digest.pipeline import RunResult
from arxiv_digest.store import Store
from arxiv_digest.webapp import create_app

TOKEN = "test-token"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ARXIV_DIGEST_HOME", str(tmp_path / "home"))
    return tmp_path


@pytest.fixture
def config_path(home):
    path = home / "config.toml"
    interests = [cfg.Interest(name="Fast sims", description="emulators", keywords=["emulator"])]
    text = cfg.render_template(["astro-ph.CO"], str(home / "digests"), interests)
    path.write_text("# my own comment\n" + text)
    return path


def _client(config_path, run=None, base_url="http://127.0.0.1:8765"):
    app = create_app(
        config_path,
        run=run
        or (
            lambda c, o, log: RunResult(
                window_start=datetime.now(UTC), window_end=datetime.now(UTC)
            )
        ),
        token=TOKEN,
    )
    return TestClient(app, base_url=base_url), app


def _save_digest():
    digest = Digest(
        generated=datetime(2026, 11, 3, 9, 30).astimezone(),
        categories=["astro-ph.CO"],
        interests=["Fast sims"],
        top_picks=3,
        new_papers=616,
        shortlisted=35,
        window_start=datetime(2026, 10, 30, tzinfo=UTC),
        window_end=datetime(2026, 11, 2, tzinfo=UTC),
        ranked_by="Chosen by Claude.",
        entries=[
            Entry(
                make_paper("2611.00001", title="Neural $N$-body emulators"),
                "Fast sims",
                why="Speeds things up.",
                relevance=9,
            )
        ],
    )
    store = Store(paths.database_path())
    try:
        return store.save_digest(digest, None, paths.data_dir() / "x.html")
    finally:
        store.close()


def test_index_lists_digests(config_path):
    digest_id = _save_digest()
    client, _ = _client(config_path)
    page = client.get("/")
    assert page.status_code == 200
    assert f'href="/digest/{digest_id}"' in page.text
    assert "Tue 3 Nov 2026, 09:30" in page.text
    assert f'name="csrf" value="{TOKEN}"' in page.text


def test_index_without_config_invites_setup(home):
    client, _ = _client(home / "missing.toml")
    assert "Set up your interests" in client.get("/").text


def test_digest_page(config_path):
    digest_id = _save_digest()
    client, _ = _client(config_path)
    page = client.get(f"/digest/{digest_id}")
    assert page.status_code == 200
    assert 'href="https://arxiv.org/abs/2611.00001"' in page.text
    assert "Speeds things up." in page.text and "katex" in page.text
    assert client.get("/digest/999").status_code == 404


def test_other_hosts_are_rejected(config_path):
    client, _ = _client(config_path, base_url="http://evil.example")
    assert client.get("/").status_code == 400


def test_forms_need_the_token(config_path):
    client, _ = _client(config_path)
    assert client.post("/settings", data={"category": "cs.LG"}).status_code == 403
    assert client.post("/run", data={"csrf": "wrong"}).status_code == 403


def test_settings_page_shows_current_settings(config_path):
    client, _ = _client(config_path)
    page = client.get("/settings").text
    assert 'value="Fast sims"' in page
    assert 'value="astro-ph.CO" checked' in page
    assert 'name="interest-1-name"' in page  # the empty "add an interest" block


def _form(**overrides):
    data = {
        "csrf": TOKEN,
        "category": ["astro-ph.CO", "cs.LG"],
        "interest-0-name": "Fast sims",
        "interest-0-description": "Neural emulators\nfor cosmology",
        "interest-0-keywords": "emulator, N-body",
        "interest-0-authors": "",
        "interest-1-name": "Lensing",
        "interest-1-keywords": "shear",
        "interest-1-exclude": "quantum",
        "output_dir": "~/arxiv-digest",
        "max_papers": "12",
        "top_picks": "2",
        "use_claude": "on",
        "claude_model": "haiku",
    }
    data.update(overrides)
    return data


def test_saving_settings(config_path):
    client, _ = _client(config_path)
    response = client.post("/settings", data=_form(), follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/settings?saved=1"
    conf = cfg.load(config_path)
    assert conf.categories == ["astro-ph.CO", "cs.LG"]
    assert [i.name for i in conf.interests] == ["Fast sims", "Lensing"]
    assert conf.interests[0].description == "Neural emulators for cosmology"
    assert conf.interests[0].keywords == ["emulator", "N-body"]
    assert conf.interests[1].exclude == ["quantum"]
    assert (conf.max_papers, conf.top_picks, conf.claude_model) == (12, 2, "haiku")
    assert "# my own comment" in config_path.read_text()
    assert "Saved." in client.get("/settings?saved=1").text


def test_removing_an_interest_and_turning_claude_off(config_path):
    client, _ = _client(config_path)
    data = _form(**{"interest-1-delete": "on"})
    del data["use_claude"]  # unchecked boxes aren't sent
    client.post("/settings", data=data)
    conf = cfg.load(config_path)
    assert [i.name for i in conf.interests] == ["Fast sims"] and conf.use_claude is False


def test_invalid_settings_are_explained_and_not_saved(config_path):
    before = config_path.read_text()
    client, _ = _client(config_path)
    page = client.post("/settings", data=_form(category=[], extra_categories="astro-ph.XX"))
    assert page.status_code == 422
    assert "Unknown arXiv category" in page.text
    assert 'value="Lensing"' in page.text  # what you typed is kept
    assert config_path.read_text() == before


def test_settings_can_create_a_config(home):
    path = home / "new" / "config.toml"
    client, _ = _client(path)
    assert "Will be created at" in client.get("/settings").text
    client.post("/settings", data=_form())
    assert [i.name for i in cfg.load(path).interests] == ["Fast sims", "Lensing"]


def test_run_in_background_and_report_progress(config_path):
    release = threading.Event()
    seen = {}

    def fake_run(conf, options, log):
        seen["options"] = options
        log("Fetching new papers…")
        release.wait(5)
        return RunResult(window_start=datetime.now(UTC), window_end=datetime.now(UTC))

    client, app = _client(config_path, run=fake_run)
    response = client.post("/run", data={"csrf": TOKEN, "use_claude": "on"}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/run"
    status = client.get("/run/status").json()
    assert status["state"] == "running" and "Fetching new papers…" in status["lines"]
    assert "Running…" in client.get("/run").text
    client.post("/run", data={"csrf": TOKEN})  # a second run doesn't start
    release.set()
    app.state.runner.thread.join(5)
    status = client.get("/run/status").json()
    assert status["state"] == "done" and status["message"] == "No new papers since the last run."
    assert seen["options"].use_claude is True and seen["options"].include_seen is False


def test_run_errors_are_reported(config_path):
    def broken(conf, options, log):
        raise RuntimeError("arXiv is down")

    client, app = _client(config_path, run=broken)
    client.post("/run", data={"csrf": TOKEN})
    app.state.runner.thread.join(5)
    status = client.get("/run/status").json()
    assert status["state"] == "error" and "arXiv is down" in status["message"]


def test_start_page_shows_interests_for_editing(config_path):
    client, _ = _client(config_path)
    page = client.get("/").text
    assert 'action="/interests"' in page
    assert "<strong>Fast sims</strong>" in page and "— emulators" in page
    assert 'name="interest-0-name" value="Fast sims"' in page
    assert "+ Add an interest" in page and 'name="interest-1-name"' in page


def _interest_form(**overrides):
    data = {
        "csrf": TOKEN,
        "interest-0-name": "Fast sims",
        "interest-0-description": "Neural emulators for N-body simulations",
        "interest-0-keywords": "emulator",
        "interest-1-name": "Lensing",
        "interest-1-keywords": "shear, intrinsic alignments",
        "action": "save",
    }
    data.update(overrides)
    return data


def test_start_page_saves_interests_and_keeps_the_rest(config_path):
    client, _ = _client(config_path)
    response = client.post("/interests", data=_interest_form(), follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/?saved=1"
    conf = cfg.load(config_path)
    assert [i.name for i in conf.interests] == ["Fast sims", "Lensing"]
    assert conf.interests[0].description == "Neural emulators for N-body simulations"
    assert conf.interests[1].keywords == ["shear", "intrinsic alignments"]
    assert conf.categories == ["astro-ph.CO"]  # untouched
    assert "# my own comment" in config_path.read_text()
    assert "Interests saved." in client.get("/?saved=1").text


def test_start_page_save_and_rerun(config_path):
    seen = {}

    def fake_run(conf, options, log):
        seen["options"] = options
        seen["interests"] = [i.name for i in conf.interests]
        return RunResult(window_start=datetime.now(UTC), window_end=datetime.now(UTC))

    client, app = _client(config_path, run=fake_run)
    response = client.post(
        "/interests", data=_interest_form(action="save_run"), follow_redirects=False
    )
    assert response.headers["location"] == "/run"
    app.state.runner.thread.join(5)
    assert seen["interests"] == ["Fast sims", "Lensing"]  # runs with the new interests
    assert seen["options"].include_seen is True


def test_start_page_remove_interest_and_errors(config_path):
    client, _ = _client(config_path)
    client.post("/interests", data=_interest_form(**{"interest-1-delete": "on"}))
    assert [i.name for i in cfg.load(config_path).interests] == ["Fast sims"]

    before = config_path.read_text()
    bad = _interest_form(**{"interest-1-keywords": "", "interest-1-name": "No signal"})
    page = client.post("/interests", data=bad)
    assert page.status_code == 422 and "needs a `description`" in page.text
    assert 'value="No signal"' in page.text  # what you typed is kept
    assert config_path.read_text() == before

    assert client.post("/interests", data={"interest-0-name": "x"}).status_code == 403
