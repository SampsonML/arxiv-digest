import threading
from datetime import UTC, datetime

import pytest
from conftest import make_paper
from starlette.testclient import TestClient

from arxiv_digest import config as cfg
from arxiv_digest import paths
from arxiv_digest.claude_cli import ClaudeStatus
from arxiv_digest.digest import Digest, Entry
from arxiv_digest.pipeline import RunResult
from arxiv_digest.store import Store
from arxiv_digest.webapp import ClaudeAccount, create_app, sign_in_links

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


class FakeAccount:
    """Stands in for Claude Code's sign-in, so tests never call the real `claude`."""

    def __init__(self, logged_in=True, installed=True):
        self.current = ClaudeStatus(
            installed=installed,
            logged_in=logged_in,
            email="me@example.com",
            plan="max",
            method="claude.ai" if logged_in else None,
        )
        self.started = 0
        self.running = False
        self.output = []

    def status(self, fresh=False):
        return self.current

    def start_login(self):
        self.started += 1
        self.running = True
        return True

    def state(self):
        return {
            **self.current.to_dict(),
            "login_running": self.running,
            "output": self.output,
            "links": sign_in_links(self.output),
        }


def _no_run(conf, options, log):
    return RunResult(window_start=datetime.now(UTC), window_end=datetime.now(UTC))


def _client(config_path, run=None, base_url="http://127.0.0.1:8765", account=None):
    app = create_app(config_path, run=run or _no_run, token=TOKEN, account=account or FakeAccount())
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


def test_digest_can_be_saved_as_html_or_markdown(config_path):
    digest_id = _save_digest()
    client, _ = _client(config_path)
    page = client.get(f"/digest/{digest_id}").text
    assert f'href="/digest/{digest_id}/save/html" download' in page
    assert f'href="/digest/{digest_id}/save/markdown" download' in page

    html = client.get(f"/digest/{digest_id}/save/html")
    assert html.status_code == 200 and html.headers["content-type"].startswith("text/html")
    assert html.headers["content-disposition"].endswith('filename="arxiv-digest-2026-11-03.html"')
    assert html.text.startswith("<!doctype html>") and "arxiv.org/abs/2611.00001" in html.text

    md = client.get(f"/digest/{digest_id}/save/markdown")
    assert md.headers["content-type"].startswith("text/markdown")
    assert md.headers["content-disposition"] == 'attachment; filename="arxiv-digest-2026-11-03.md"'
    assert "[arXiv:2611.00001](https://arxiv.org/abs/2611.00001)" in md.text

    assert client.get(f"/digest/{digest_id}/save/pdf").status_code == 404
    assert client.get("/digest/999/save/html").status_code == 404


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
    assert "Deep reads use sonnet" in page


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
    assert 'class="chip" data-edit="0" style="--tag: #e3925a"' in page
    assert 'title="emulators"' in page and ">Fast sims</button>" in page
    assert 'data-panel="0"' in page and 'name="interest-0-name" value="Fast sims"' in page
    assert (
        "+ Add interest" in page and 'data-panel="new"' in page and 'name="interest-1-name"' in page
    )


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


def test_start_page_shows_claude_sign_in_status(config_path):
    client, _ = _client(config_path)
    page = client.get("/").text
    assert "✓ Signed in to Claude as me@example.com (Max plan)" in page
    assert ">Log in to a Claude account</button>" in page and 'action="/claude/login"' in page

    client, _ = _client(config_path, account=FakeAccount(logged_in=False))
    page = client.get("/").text
    assert "not signed in to a Claude account" in page and "never API keys" in page
    assert ">Log in to your Claude account</button>" in page

    client, _ = _client(config_path, account=FakeAccount(installed=False, logged_in=None))
    assert "Install Claude Code" in client.get("/").text


def test_settings_links_to_sign_in_instead_of_nesting_a_form(config_path):
    client, _ = _client(config_path, account=FakeAccount(logged_in=False))
    page = client.get("/settings").text
    assert '<a class="button primary" href="/claude">Log in to your Claude account</a>' in page
    assert 'action="/claude/login"' not in page
    client, _ = _client(config_path)
    page = client.get("/settings").text
    assert '<a class="button small" href="/claude">Log in to a Claude account</a>' in page
    assert 'action="/claude/login"' not in page


def test_sign_in_button_starts_claude_login(config_path):
    account = FakeAccount(logged_in=False)
    client, _ = _client(config_path, account=account)
    assert client.post("/claude/login", data={}).status_code == 403
    assert account.started == 0
    response = client.post("/claude/login", data={"csrf": TOKEN}, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/claude"
    assert account.started == 1
    account.output = ["Opening https://claude.ai/oauth/authorize?code=1 in your browser"]
    page = client.get("/claude").text
    assert "Finish logging in" in page
    assert 'href="https://claude.ai/oauth/authorize?code=1"' in page
    status = client.get("/claude/status").json()
    assert status["login_running"] is True and status["logged_in"] is False


def test_sign_in_page_when_already_signed_in(config_path):
    client, _ = _client(config_path)
    page = client.get("/claude").text
    assert "Signed in to Claude as me@example.com" in page
    assert ">Log in to a different Claude account</button>" in page


def test_sign_in_links_only_point_to_claude():
    lines = [
        "Visit https://claude.ai/oauth/authorize?state=abc to sign in",
        "or https://console.anthropic.com/login",
        "not https://evil.example/claude.ai or https://claude.ai.evil.example/x",
    ]
    assert sign_in_links(lines) == [
        "https://claude.ai/oauth/authorize?state=abc",
        "https://console.anthropic.com/login",
    ]


class FakeProcess:
    def __init__(self, lines):
        self.stdout = iter(lines)
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


def test_claude_account_runs_login_in_the_background():
    import subprocess
    import sys
    import time

    from arxiv_digest.claude_cli import ClaudeCLI

    started = []

    def popen(argv, **kwargs):
        started.append((argv, kwargs))
        return FakeProcess(["Opening https://claude.ai/oauth/authorize?x=1\n", "Done\n"])

    def status_runner(argv, **kwargs):
        out = '{"loggedIn": true, "authMethod": "claude.ai", "email": "a@b.c"}'
        return subprocess.CompletedProcess(argv, 0, out, "")

    account = ClaudeAccount(lambda: ClaudeCLI(sys.executable, runner=status_runner), popen=popen)
    assert account.start_login() is True
    argv, kwargs = started[0]
    assert argv[-3:] == ["auth", "login", "--claudeai"] and kwargs["stdin"] == subprocess.DEVNULL
    assert "ANTHROPIC_API_KEY" not in kwargs["env"]
    for _ in range(200):  # the output is read by a background thread
        if account.process.poll() is not None:
            break
        time.sleep(0.01)
    state = account.state()
    assert state["logged_in"] is True and state["ready"] is True and state["email"] == "a@b.c"
    assert state["links"] == ["https://claude.ai/oauth/authorize?x=1"]
    assert state["output"][-1] == "Done"


def _with_background(config_path, value):
    text = config_path.read_text().replace(
        '# app_background = "default"', f"app_background = {value!r}".replace("'", '"')
    )
    config_path.write_text(text)


def test_background_image_is_served_and_used(config_path):
    client, _ = _client(config_path)
    page = client.get("/").text
    assert '<body class="has-bg"' in page
    url = page.split("--bg-image: url('", 1)[1].split("'", 1)[0]
    assert url.startswith("/background?v=")  # versioned, so a new image is never cached stale
    image = client.get(url)
    assert "immutable" in image.headers["cache-control"]
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
    assert image.content[:2] == b"\xff\xd8"  # a JPEG


def test_background_can_be_turned_off_or_replaced(config_path, tmp_path):
    _with_background(config_path, "none")
    client, _ = _client(config_path)
    assert "<body>" in client.get("/").text
    assert client.get("/background").status_code == 404

    own = tmp_path / "mine.png"
    own.write_bytes(b"\x89PNG fake")
    config_path.write_text(config_path.read_text().replace('"none"', f'"{own}"'))
    response = client.get("/background")
    assert response.headers["content-type"] == "image/png" and response.content == own.read_bytes()

    own.unlink()  # a missing custom image falls back to the bundled one
    assert client.get("/background").headers["content-type"] == "image/jpeg"


def test_start_page_header_uses_the_latest_digest(config_path):
    client, _ = _client(config_path)
    page = client.get("/").text
    assert "No digests yet" in page and "Good " in page
    digest_id = _save_digest()
    page = client.get("/").text
    assert '<span class="num">616</span>' in page and '<span class="num">35</span>' in page
    assert "picked 1 of 616 new papers" in page
    assert f'href="/digest/{digest_id}">Open latest digest' in page


def test_fonts_are_served_safely(config_path):
    client, _ = _client(config_path)
    font = client.get("/fonts/inter-latin.woff2")
    assert font.status_code == 200 and font.headers["content-type"] == "font/woff2"
    assert font.content[:4] == b"wOF2"
    assert client.get("/fonts/..%2Fbackground.jpg").status_code == 404
    assert client.get("/fonts/missing.woff2").status_code == 404
    assert (
        "@font-face" in client.get("/").text
        and 'font-family: "Space Grotesk"' in client.get("/").text
    )


def test_run_tracker_reports_steps_and_counts(config_path):
    release = threading.Event()

    def staged_run(conf, options, log):
        log.step("fetch", {"done": 500, "total": 925})
        log.step("shortlist", {"new": 616})
        log.step("claude", {"shortlisted": 35})
        release.wait(5)
        log.step("write", {"picked": 3})
        return RunResult(window_start=datetime.now(UTC), window_end=datetime.now(UTC))

    client, app = _client(config_path, run=staged_run)
    client.post("/run", data={"csrf": TOKEN, "use_claude": "on"})
    for _ in range(100):
        status = client.get("/run/status").json()
        if status["step"] == "claude":
            break
        threading.Event().wait(0.01)
    assert status["state"] == "running" and status["claude_used"] is True
    assert status["counts"] == {"fetched": 500, "total": 925, "new": 616, "shortlisted": 35}
    page = client.get("/run").text
    assert 'id="tracker"' in page and '"step": "claude"' in page
    release.set()
    app.state.runner.thread.join(5)
    status = client.get("/run/status").json()
    assert status["state"] == "done" and status["counts"]["picked"] == 0  # from the result


def test_greeting_and_time_ago():
    from datetime import timedelta

    from arxiv_digest.webapp import greeting, time_ago

    morning = datetime(2026, 11, 3, 8, 0)
    assert greeting(morning) == "Good morning"
    assert greeting(morning.replace(hour=14)) == "Good afternoon"
    assert greeting(morning.replace(hour=22)) == "Good evening"
    assert time_ago(morning - timedelta(seconds=30), morning) == "just now"
    assert time_ago(morning - timedelta(minutes=20), morning) == "20 minutes ago"
    assert time_ago(morning - timedelta(hours=5), morning) == "5 hours ago"
    assert time_ago(morning - timedelta(days=1, hours=2), morning) == "yesterday"
    assert time_ago(morning - timedelta(days=4), morning) == "4 days ago"


def test_health_heartbeat_and_icon(config_path):
    client, _ = _client(config_path)
    assert client.get("/health").json()["app"] == "arxiv-digest"
    assert client.get("/heartbeat").status_code == 204
    icon = client.get("/icon.svg")
    assert icon.headers["content-type"].startswith("image/svg+xml") and b"<svg" in icon.content
    page = client.get("/").text
    assert 'rel="icon"' in page and 'action="/quit"' in page and "/heartbeat" in page


def test_quit_stops_the_server_after_responding(config_path):
    stopped = threading.Event()
    client, app = _client(config_path)
    app.state.on_quit = stopped.set
    assert client.post("/quit", data={}).status_code == 403
    page = client.post("/quit", data={"csrf": TOKEN})
    assert page.status_code == 200 and "arxiv-digest has stopped" in page.text
    assert 'action="/quit"' not in page.text  # no nav on the stopped page
    assert stopped.wait(3)


def test_quit_waits_for_a_run(config_path):
    release = threading.Event()

    def slow_run(conf, options, log):
        release.wait(5)
        return RunResult(window_start=datetime.now(UTC), window_end=datetime.now(UTC))

    client, app = _client(config_path, run=slow_run)
    app.state.on_quit = lambda: None
    client.post("/run", data={"csrf": TOKEN})
    page = client.post("/quit", data={"csrf": TOKEN})
    assert page.status_code == 409 and "in progress" in page.text
    release.set()
    app.state.runner.thread.join(5)


def test_idle_rules_and_watchdog(config_path):
    from arxiv_digest.webapp import should_stop

    assert should_stop(700, 600, busy=False)
    assert not should_stop(100, 600, busy=False)
    assert not should_stop(10_000, 600, busy=True)

    stopped = threading.Event()
    app = create_app(config_path, run=_no_run, token=TOKEN, account=FakeAccount(), idle_exit=1)
    app.state.on_quit = stopped.set
    assert stopped.wait(5)  # nobody is using it, so it asks to stop


def test_health_checks_do_not_count_as_activity(config_path):
    client, app = _client(config_path)
    activity = app.state.activity
    activity.last = activity.clock() - 500
    client.get("/health")
    assert activity.idle_for() >= 500
    client.get("/heartbeat")
    assert activity.idle_for() < 5


def test_deep_read_option_step_and_setting(config_path):
    seen = {}

    def run_with_deep(conf, options, log):
        seen["deep"] = options.deep_read
        log.step("claude", {"shortlisted": 35})
        log.step("deep_read", {"papers": 3})
        return RunResult(window_start=datetime.now(UTC), window_end=datetime.now(UTC))

    client, app = _client(config_path, run=run_with_deep)
    assert 'name="deep_read" checked' in client.get("/").text
    client.post("/run", data={"csrf": TOKEN, "use_claude": "on"})  # deep read box unticked
    app.state.runner.thread.join(5)
    status = client.get("/run/status").json()
    assert seen["deep"] is False
    assert status["deep_used"] is True and status["counts"]["deep"] == 3
    assert 'data-step="deep_read"' in client.get("/run").text

    page = client.get("/settings").text
    assert 'name="deep_reads"' in page and 'value="3"' in page
    client.post("/settings", data=_form(deep_reads="1"))
    assert cfg.load(config_path).deep_reads == 1


def test_offer_to_add_the_launcher_entry(config_path, monkeypatch):
    from arxiv_digest import launcher

    installed = []
    monkeypatch.setattr(launcher, "shortcut_installed", lambda: bool(installed))
    monkeypatch.setattr(launcher, "shortcut_place", lambda: "Applications")
    monkeypatch.setattr(launcher, "install_shortcut", lambda config=None: installed.append(config))
    client, _ = _client(config_path)
    page = client.get("/").text
    assert ">Add to Applications</button>" in page and "No thanks" in page
    assert client.post("/shortcut", data={"action": "install"}).status_code == 403
    response = client.post(
        "/shortcut", data={"csrf": TOKEN, "action": "install"}, follow_redirects=False
    )
    assert response.headers["location"] == "/?shortcut=added" and len(installed) == 1
    assert "Open arxiv-digest from Applications" in client.get("/?shortcut=added").text
    assert "Add to Applications" not in client.get("/").text  # installed: no more offer


def test_no_thanks_is_remembered(config_path, monkeypatch):
    from arxiv_digest import launcher

    monkeypatch.setattr(launcher, "shortcut_installed", lambda: False)
    client, _ = _client(config_path)
    assert 'class="card shortcut-offer"' in client.get("/").text
    client.post("/shortcut", data={"csrf": TOKEN, "action": "dismiss"})
    assert 'class="card shortcut-offer"' not in client.get("/").text
