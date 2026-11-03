"""The local web app: browse digests, run a new one, and edit settings.

`arxiv-digest app` serves it on 127.0.0.1 only. Two protections matter for a
local app that can change files and start runs:
  - Host check: requests must be addressed to 127.0.0.1 or localhost, which
    blocks DNS-rebinding attacks from web pages open in the same browser;
  - CSRF token: every form carries a random per-session token, so another web
    site cannot submit forms to the app on your behalf.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import subprocess
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from . import __version__, launcher, paths
from . import categories as cats
from . import config as cfg
from .claude_cli import SIGN_IN_HINT, ClaudeCLI, ClaudeStatus
from .html_render import (
    KATEX_HEAD,
    KATEX_SCRIPTS,
    digest_body,
    interest_colors,
    standalone_page,
    stylesheet,
)
from .pipeline import RunOptions, RunResult, run_with_config
from .render import render_markdown
from .store import Store

MODEL_CHOICES = ["sonnet", "haiku", "opus"]
SAVE_FORMATS = {  # the digest page's Save menu: kind -> (file suffix, media type)
    "html": ("html", "text/html; charset=utf-8"),
    "markdown": ("md", "text/markdown; charset=utf-8"),
}
RunFunction = Callable[[cfg.Config, RunOptions, Callable[[str], None]], RunResult]


def _default_run(config: cfg.Config, options: RunOptions, log: Callable[[str], None]) -> RunResult:
    progress = getattr(log, "step", None)  # the Runner's reporter tracks steps too
    if options.mode == "weird":
        from .weird import run_weird_with_config

        run = run_weird_with_config
    else:
        run = run_with_config
    if progress is None:
        return run(config, options, log=log)
    return run(config, options, log=log, progress=progress)


# ---------------------------------------------------------------------------
# Background runs


@dataclass
class Job:
    state: str = "idle"  # idle | running | done | error
    lines: list[str] = field(default_factory=list)
    digest_id: int | None = None
    message: str = ""
    started: datetime | None = None
    step: str = ""  # fetch | shortlist | claude | write (see pipeline.Progress)
    counts: dict[str, int | None] = field(default_factory=dict)
    claude_used: bool = False
    deep_used: bool = False
    mode: str = "digest"  # or "weird" (Find something weird)


class _Reporter:
    """The `log` handed to a run: collects log lines and tracks the current step."""

    def __init__(self, job: Job) -> None:
        self.job = job

    def __call__(self, line: str) -> None:
        self.job.lines.append(line)

    def step(self, name: str, details: dict) -> None:
        job = self.job
        job.step = name
        if name == "fetch":
            job.counts["fetched"] = details.get("done")
            job.counts["total"] = details.get("total")
        elif name == "shortlist":
            job.counts["new"] = details.get("new")
        elif name == "claude":
            job.counts["shortlisted"] = details.get("shortlisted")
            job.claude_used = True
        elif name == "deep_read":
            job.counts["deep"] = details.get("papers")
            job.deep_used = True
        elif name == "write":
            job.counts["picked"] = details.get("picked")


class Runner:
    """Runs one digest at a time in a background thread, collecting its log."""

    def __init__(self, run: RunFunction) -> None:
        self._run = run
        self._lock = threading.Lock()
        self.job = Job()
        self.thread: threading.Thread | None = None

    def start(self, config: cfg.Config, options: RunOptions) -> bool:
        with self._lock:
            if self.job.state == "running":
                return False
            self.job = Job(state="running", started=datetime.now(), mode=options.mode)
            job = self.job
        self.thread = threading.Thread(target=self._work, args=(job, config, options), daemon=True)
        self.thread.start()
        return True

    def _work(self, job: Job, config: cfg.Config, options: RunOptions) -> None:
        try:
            result = self._run(config, options, _Reporter(job))
        except Exception as exc:  # shown in the page; the server keeps running
            job.lines.append("".join(traceback.format_exception_only(exc)).strip())
            job.message = str(exc) or exc.__class__.__name__
            job.state = "error"
            return
        job.lines.extend(f"Note: {note}" for note in result.notes)
        job.digest_id = result.digest_id
        job.counts.update(
            fetched=result.fetched,
            new=len(result.new_papers),
            shortlisted=len(result.shortlisted),
            picked=len(result.picks),
        )
        if result.digest is None:
            job.message = (
                "No papers in the latest announcement."
                if job.mode == "weird"
                else "No new papers since the last run."
            )
        elif job.mode == "weird":
            job.message = (
                f"{len(result.new_papers)} papers · {len(result.shortlisted)} most unusual"
                f" · {len(result.picks)} weird ones picked"
            )
        else:
            job.message = (
                f"{len(result.new_papers)} new papers · {len(result.shortlisted)} shortlisted"
                f" · {len(result.picks)} picked"
            )
        job.state = "done"


# ---------------------------------------------------------------------------
# Claude Code sign-in

SIGN_IN_HOSTS = ("claude.ai", "claude.com", "anthropic.com")
_URL = re.compile(r"https://[^\s'\"<>]+")


def sign_in_links(lines: list[str]) -> list[str]:
    """Sign-in URLs printed by `claude auth login`, limited to Anthropic's own sites."""
    links: list[str] = []
    for line in lines:
        for url in _URL.findall(line):
            host = (urlparse(url).hostname or "").lower()
            if any(host == h or host.endswith("." + h) for h in SIGN_IN_HOSTS) and url not in links:
                links.append(url)
    return links


class ClaudeAccount:
    """Claude Code's sign-in status (cached briefly) and a background `claude auth login`.

    The login itself is Claude Code's: it opens Claude's sign-in page in the
    browser and stores the credentials where Claude Code keeps them.
    arxiv-digest never sees them.
    """

    LOGIN_TIMEOUT = 600  # seconds before an unfinished sign-in is stopped

    def __init__(
        self,
        cli_factory: Callable[[], ClaudeCLI],
        *,
        popen: Callable[..., Any] = subprocess.Popen,
        cache_seconds: float = 10.0,
    ) -> None:
        self._cli_factory = cli_factory
        self._popen = popen
        self._cache_seconds = cache_seconds
        self._cached: tuple[float, ClaudeStatus] | None = None
        self._lock = threading.Lock()
        self.process: Any = None
        self.output: list[str] = []

    def status(self, fresh: bool = False) -> ClaudeStatus:
        now = time.monotonic()
        if not fresh and self._cached and now - self._cached[0] < self._cache_seconds:
            return self._cached[1]
        status = self._cli_factory().status()
        self._cached = (now, status)
        return status

    @property
    def login_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start_login(self) -> bool:
        """Start `claude auth login` unless it is already running; False if not installed."""
        cli = self._cli_factory()
        if not cli.available():
            return False
        with self._lock:
            if self.login_running:
                return True
            self.output = []
            self.process = self._popen(
                cli.login_command(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=cli.env,  # no API keys: this is a Claude account sign-in
            )
            process = self.process
        threading.Thread(target=self._follow, args=(process,), daemon=True).start()
        return True

    def _follow(self, process: Any) -> None:
        timer = threading.Timer(self.LOGIN_TIMEOUT, process.kill)
        timer.daemon = True
        timer.start()
        try:
            for line in process.stdout or []:
                self.output.append(line.rstrip())
            process.wait()
        finally:
            timer.cancel()
            self._cached = None  # the next status check asks Claude Code again

    def state(self) -> dict[str, Any]:
        status = self.status(fresh=True)
        return {
            **status.to_dict(),
            "login_running": self.login_running,
            "output": self.output[-20:],
            "links": sign_in_links(self.output),
        }


# ---------------------------------------------------------------------------
# Interest forms (on the start page and in Settings)

INTEREST_FIELDS = ("description", "keywords", "require", "exclude", "authors")


def interests_from_form(form: Any) -> list[cfg.Interest]:
    """Interests from fields named interest-<n>-<field>, skipping removed and empty ones."""
    interests = []
    indexes = sorted(
        {
            int(k.split("-")[1])
            for k in form
            if k.startswith("interest-") and k.count("-") >= 2 and k.split("-")[1].isdigit()
        }
    )
    for i in indexes:
        if form.get(f"interest-{i}-delete"):
            continue
        name = str(form.get(f"interest-{i}-name", "")).strip()
        fields = {key: str(form.get(f"interest-{i}-{key}", "")) for key in INTEREST_FIELDS}
        if not name and not any(v.strip() for v in fields.values()):
            continue  # the empty "add an interest" block
        interests.append(
            cfg.Interest(
                name=name,
                description=" ".join(fields["description"].split()),
                keywords=_split(fields["keywords"]),
                require=_split(fields["require"]),
                exclude=_split(fields["exclude"]),
                authors=_split(fields["authors"]),
            )
        )
    return interests


def interest_summary(interest: cfg.Interest, limit: int = 90) -> str:
    """One line describing an interest, for its collapsed row on the start page."""
    if interest.description:
        text = interest.description
    elif interest.keywords:
        text = "keywords: " + ", ".join(interest.keywords)
    elif interest.authors:
        text = "authors: " + ", ".join(interest.authors)
    else:
        text = ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


EMPTY_INTEREST = cfg.Interest(name="")


def greeting(now: datetime) -> str:
    hour = now.hour
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 18:
        return "Good afternoon"
    return "Good evening"


def time_ago(then: datetime, now: datetime) -> str:
    seconds = max(0, (now - then).total_seconds())
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} minutes ago"
    if seconds < 2 * 3600:
        return "an hour ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} hours ago"
    days = int(seconds // 86400)
    return "yesterday" if days == 1 else f"{days} days ago"


FONT_FILE = re.compile(r"^[a-z-]+\.woff2$")


# ---------------------------------------------------------------------------
# Settings form


def _split(text: str) -> list[str]:
    return [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]


def _category_groups(selected: set[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[str, str]]] = {}
    for code, name in sorted(cats.CATEGORIES.items()):
        archive = code.split(".")[0] if "." in code else "physics (other)"
        groups.setdefault(archive, []).append((code, name))
    return [
        {
            "name": archive,
            "items": [{"code": c, "name": n, "checked": c in selected} for c, n in items],
            "count": sum(c in selected for c, _ in items),
        }
        for archive, items in sorted(groups.items())
    ]


@dataclass
class SettingsForm:
    categories: list[str]
    interests: list[cfg.Interest]
    values: dict[str, Any]

    @classmethod
    def from_config(cls, conf: cfg.Config | None) -> SettingsForm:
        if conf is None:
            return cls(
                categories=[],
                interests=[],
                values={
                    "max_papers": 15,
                    "top_picks": 3,
                    "use_claude": True,
                    "claude_model": "sonnet",
                    "deep_reads": 3,
                },
            )
        return cls(
            categories=list(conf.categories),
            interests=list(conf.interests),
            values={
                "max_papers": conf.max_papers,
                "top_picks": conf.top_picks,
                "use_claude": conf.use_claude,
                "claude_model": conf.claude_model,
                "deep_reads": conf.deep_reads,
            },
        )

    @classmethod
    async def from_request(cls, request: Request) -> SettingsForm:
        form = await request.form()
        categories = [str(c) for c in form.getlist("category")]
        categories += _split(str(form.get("extra_categories", "")))
        interests = interests_from_form(form)

        def number(key: str, default: int) -> int | str:
            raw = str(form.get(key, "")).strip()
            try:
                return int(raw)
            except ValueError:
                return raw or default  # a non-number makes validation explain the problem

        return cls(
            categories=categories,
            interests=interests,
            values={
                "max_papers": number("max_papers", 15),
                "top_picks": number("top_picks", 3),
                "use_claude": form.get("use_claude") == "on",
                "claude_model": str(form.get("claude_model", "sonnet")).strip() or "sonnet",
                "deep_reads": number("deep_reads", 3),
            },
        )


# ---------------------------------------------------------------------------
# Stopping when nobody is using the app


class Activity:
    """When the app last received a request (pages, polling, heartbeats)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.last = clock()

    def touch(self) -> None:
        self.last = self.clock()

    def idle_for(self) -> float:
        return self.clock() - self.last


class ActivityTracker:
    """ASGI middleware that records every HTTP request as activity."""

    def __init__(self, app: Any, activity: Activity) -> None:
        self.app = app
        self.activity = activity

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        # Health checks (from launchers or monitoring) don't count as someone using the app.
        if scope["type"] == "http" and scope.get("path") != "/health":
            self.activity.touch()
        await self.app(scope, receive, send)


def should_stop(idle_seconds: float, limit: float, busy: bool) -> bool:
    return not busy and idle_seconds >= limit


def _idle_watchdog(
    app: Any,
    activity: Activity,
    limit: float,
    busy: Callable[[], bool],
    stop: Callable[[], None],
) -> None:
    check_every = max(1.0, min(30.0, limit / 4))
    while not app.state.stopping:
        time.sleep(check_every)
        if busy():
            activity.touch()  # a long run counts as use
        elif should_stop(activity.idle_for(), limit, False):
            stop()
            return


# ---------------------------------------------------------------------------
# The app


def create_app(
    config_path: Path | None = None,
    *,
    run: RunFunction = _default_run,
    token: str | None = None,
    account: ClaudeAccount | None = None,
    idle_exit: float | None = None,
) -> Starlette:
    """The app. With `idle_exit` (seconds), it asks to stop once no tab has been
    open for that long (open tabs send a heartbeat every minute); the server
    supplies the stop function as `app.state.on_quit`."""
    config_path = config_path or paths.config_path()
    csrf_token = token or secrets.token_urlsafe(32)
    runner = Runner(run)
    activity = Activity()

    def current_cli() -> ClaudeCLI:
        conf, _ = load_config()
        return ClaudeCLI(conf.claude_command if conf else "claude")

    account = account or ClaudeAccount(current_cli)
    env = Environment(
        loader=PackageLoader("arxiv_digest", "templates"),
        autoescape=select_autoescape(["html"]),
    )
    assets = resources.files("arxiv_digest").joinpath("assets")
    app_css = assets.joinpath("app.css").read_text("utf-8")
    fonts_css = assets.joinpath("fonts.css").read_text("utf-8")

    bundled_background = (
        resources.files("arxiv_digest").joinpath("assets/background.jpg").read_bytes()
    )
    bundled_version = hashlib.sha256(bundled_background).hexdigest()[:12]

    def load_config() -> tuple[cfg.Config | None, str | None]:
        if not config_path.exists():
            return None, None
        try:
            return cfg.load(config_path), None
        except cfg.ConfigError as exc:
            return None, str(exc)

    def background_setting() -> str:
        conf, _ = load_config()
        return conf.app_background if conf else "default"

    def background_url() -> str | None:
        """The image's URL, versioned by its content so browsers never show a stale one."""
        setting = background_setting()
        if setting == "none":
            return None
        version = bundled_version
        if setting != "default":
            try:
                info = Path(setting).stat()
                version = f"{int(info.st_mtime)}-{info.st_size}"
            except OSError:
                pass  # missing: the bundled image is served instead
        return f"/background?v={version}"

    def page(name: str, request: Request, status: int = 200, **context: Any) -> HTMLResponse:
        template = env.get_template(name)
        html = template.render(
            background_url=background_url(),
            request=request,
            css=Markup(fonts_css + "\n" + stylesheet() + "\n" + app_css),
            csrf=csrf_token,
            version=__version__,
            job=runner.job,
            sign_in_hint=SIGN_IN_HINT,
            **context,
        )
        return HTMLResponse(html, status_code=status)

    async def check_csrf(request: Request) -> bool:
        form = await request.form()
        return secrets.compare_digest(str(form.get("csrf", "")), csrf_token)

    def render_index(
        request: Request,
        status: int = 200,
        *,
        interests: list[cfg.Interest] | None = None,
        interests_error: str | None = None,
    ) -> HTMLResponse:
        conf, problem = load_config()
        store = Store(paths.database_path())
        try:
            rows = store.list_digests()
        finally:
            store.close()
        digests = [
            {
                "id": row["id"],
                "when": datetime.fromisoformat(row["generated"]),
                "picked": row["picked"],
                "new": row["new"],
                "markdown_path": row["markdown_path"],
                "html_path": row["html_path"],
                "weird": row["kind"] == "weird",
            }
            for row in rows
        ]
        latest_row = next((row for row in rows if row["kind"] != "weird"), None)
        now = datetime.now().astimezone()
        hero: dict[str, Any] = {
            "greeting": greeting(now),
            "today": f"{now:%A} {now.day} {now:%B}",
            "latest": None,
        }
        if latest_row is not None:  # the header shows your latest interest digest
            store = Store(paths.database_path())
            try:
                found = store.get_digest(latest_row["id"])
            finally:
                store.close()
            if found:
                latest, _ = found
                hero["latest"] = {
                    "id": latest_row["id"],
                    "new": latest.new_papers,
                    "shortlisted": latest.shortlisted,
                    "picked": len(latest.entries),
                    "when_text": time_ago(latest.generated, now),
                }
        shown = interests if interests is not None else (conf.interests if conf else [])
        return page(
            "index.html",
            request,
            status,
            conf=conf,
            problem=problem,
            digests=digests,
            hero=hero,
            colors=interest_colors([i.name for i in shown]),
            interests=interests if interests is not None else (conf.interests if conf else []),
            interests_error=interests_error,
            interests_saved=request.query_params.get("saved") == "1",
            summary=interest_summary,
            empty_interest=EMPTY_INTEREST,
            claude=account.status() if conf and conf.use_claude else None,
            shortcut=shortcut_offer(request),
        )

    def shortcut_offer(request: Request) -> dict[str, Any] | None:
        added = request.query_params.get("shortcut") == "added"
        if added:
            return {"added": True, "place": launcher.shortcut_place()}
        if launcher.shortcut_installed() or launcher.offer_dismissed():
            return None
        return {"added": False, "place": launcher.shortcut_place(), "error": None}

    async def index(request: Request) -> Response:
        return render_index(request)

    async def shortcut(request: Request) -> Response:
        if not await check_csrf(request):
            return Response("Invalid or missing form token; reload the page.", status_code=403)
        form = await request.form()
        if form.get("action") == "dismiss":
            launcher.dismiss_offer()
            return RedirectResponse("/", status_code=303)
        explicit = None if config_path == paths.config_path() else config_path.resolve()
        try:
            launcher.install_shortcut(explicit)
        except (launcher.LauncherError, OSError) as exc:
            return page(
                "message.html",
                request,
                500,
                title="Couldn't add arxiv-digest",
                message=f"{exc}. You can also run `arxiv-digest shortcut install` in a terminal.",
            )
        return RedirectResponse("/?shortcut=added", status_code=303)

    async def save_interests(request: Request) -> Response:
        if not await check_csrf(request):
            return Response("Invalid or missing form token; reload the page.", status_code=403)
        if not config_path.exists():
            return RedirectResponse("/settings", status_code=303)
        form = await request.form()
        interests = interests_from_form(form)
        try:
            conf = cfg.save_interests(config_path, interests)
        except cfg.ConfigError as exc:
            return render_index(request, 422, interests=interests, interests_error=str(exc))
        if form.get("action") == "save_run":
            # Changed interests usually mean "show me today's papers again".
            runner.start(conf, RunOptions(include_seen=True, use_claude=conf.use_claude))
            return RedirectResponse("/run", status_code=303)
        return RedirectResponse("/?saved=1", status_code=303)

    async def show_digest(request: Request) -> Response:
        store = Store(paths.database_path())
        try:
            found = store.get_digest(int(request.path_params["digest_id"]))
            newer = older = None
            if found:
                ids = [row["id"] for row in store.list_digests(limit=100000)]
                position = ids.index(found[1]["id"])
                newer = ids[position - 1] if position > 0 else None
                older = ids[position + 1] if position + 1 < len(ids) else None
        finally:
            store.close()
        if found is None:
            return page(
                "message.html",
                request,
                404,
                title="Not found",
                message="There is no digest with that number.",
            )
        digest, row = found
        return page(
            "digest.html",
            request,
            digest=digest,
            body=Markup(digest_body(digest)),
            row=row,
            newer=newer,
            older=older,
            katex_head=Markup(KATEX_HEAD),
            katex_scripts=Markup(KATEX_SCRIPTS),
        )

    async def save_digest(request: Request) -> Response:
        """The Save button: the digest as a downloaded HTML or Markdown file."""
        kind = request.path_params["kind"]
        if kind not in SAVE_FORMATS:
            return Response("Unknown format", status_code=404)
        store = Store(paths.database_path())
        try:
            found = store.get_digest(int(request.path_params["digest_id"]))
        finally:
            store.close()
        if found is None:
            return Response("There is no digest with that number.", status_code=404)
        digest = found[0]
        suffix, media_type = SAVE_FORMATS[kind]
        content = standalone_page(digest) if kind == "html" else render_markdown(digest)
        filename = f"arxiv-digest-{digest.generated:%Y-%m-%d}.{suffix}"
        return Response(
            content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    async def start_run(request: Request) -> Response:
        if not await check_csrf(request):
            return Response("Invalid or missing form token; reload the page.", status_code=403)
        conf, problem = load_config()
        if conf is None:
            return RedirectResponse("/settings", status_code=303)
        form = await request.form()
        options = RunOptions(
            include_seen=form.get("include_seen") == "on",
            use_claude=form.get("use_claude") == "on",  # unchecked boxes aren't sent
            deep_read=form.get("deep_read") == "on",
        )
        runner.start(conf, options)  # if one is already running, just show it
        return RedirectResponse("/run", status_code=303)

    async def start_weird(request: Request) -> Response:
        """Find something weird: the 5 most unusual papers in the latest announcement."""
        if not await check_csrf(request):
            return Response("Invalid or missing form token; reload the page.", status_code=403)
        conf, problem = load_config()
        if conf is None:
            return RedirectResponse("/settings", status_code=303)
        form = await request.form()
        options = RunOptions(
            mode="weird",
            use_claude=form.get("use_claude") == "on",
            deep_read=form.get("deep_read") == "on",
        )
        runner.start(conf, options)
        return RedirectResponse("/run", status_code=303)

    def job_status() -> dict[str, Any]:
        job = runner.job
        return {
            "state": job.state,
            "step": job.step,
            "counts": dict(job.counts),
            "claude_used": job.claude_used,
            "deep_used": job.deep_used,
            "mode": job.mode,
            "lines": list(job.lines),
            "digest_id": job.digest_id,
            "message": job.message,
        }

    async def run_page(request: Request) -> Response:
        return page("run.html", request, job_status=job_status())

    async def run_status(request: Request) -> Response:
        return JSONResponse(job_status())

    async def health(request: Request) -> Response:
        return JSONResponse({"app": "arxiv-digest", "version": __version__})

    async def heartbeat(request: Request) -> Response:
        return Response(status_code=204)  # the activity tracker has noted the request

    icon_svg = assets.joinpath("icons", "arxiv-digest.svg").read_bytes()

    async def icon(request: Request) -> Response:
        return Response(
            icon_svg, media_type="image/svg+xml", headers={"Cache-Control": "max-age=86400"}
        )

    def busy() -> bool:
        return runner.job.state == "running" or getattr(account, "login_running", False)

    def stop_soon() -> None:
        """Stop after the current response has been sent."""
        quit_hook = getattr(app.state, "on_quit", None)
        app.state.stopping = True
        if quit_hook is not None:
            timer = threading.Timer(0.5, quit_hook)
            timer.daemon = True
            timer.start()

    async def quit_app(request: Request) -> Response:
        if not await check_csrf(request):
            return Response("Invalid or missing form token; reload the page.", status_code=403)
        if busy():
            return page(
                "message.html",
                request,
                409,
                title="Still working",
                message=(
                    "A run (or a Claude sign-in) is in progress. Quit again when it has finished."
                ),
            )
        stop_soon()
        return page("stopped.html", request)

    async def font(request: Request) -> Response:
        name = request.path_params["name"]
        if not FONT_FILE.match(name):
            return Response(status_code=404)
        file = assets.joinpath("fonts", name)
        if not file.is_file():
            return Response(status_code=404)
        return Response(
            file.read_bytes(),
            media_type="font/woff2",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    async def background(request: Request) -> Response:
        setting = background_setting()
        # The page always asks for /background?v=<content version>, so a cached
        # copy can be kept indefinitely: a changed image gets a new URL.
        headers = {"Cache-Control": "public, max-age=31536000, immutable"}
        if setting == "none":
            return Response(status_code=404)
        if setting == "default":
            return Response(bundled_background, media_type="image/jpeg", headers=headers)
        path = Path(setting)
        try:
            data = path.read_bytes()
        except OSError:  # a missing custom image falls back to the bundled one
            return Response(bundled_background, media_type="image/jpeg", headers=headers)
        media = cfg.BACKGROUND_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return Response(data, media_type=media, headers=headers)

    async def claude_status(request: Request) -> Response:
        return JSONResponse(account.state())

    async def claude_login_page(request: Request) -> Response:
        return page("claude.html", request, state=account.state())

    async def claude_login(request: Request) -> Response:
        if not await check_csrf(request):
            return Response("Invalid or missing form token; reload the page.", status_code=403)
        account.start_login()
        return RedirectResponse("/claude", status_code=303)

    def current_deep_read_model() -> str:
        conf, _ = load_config()
        return conf.deep_read_model if conf else cfg.DEEP_READ_MODEL

    async def settings_page(request: Request) -> Response:
        conf, problem = load_config()
        form = SettingsForm.from_config(conf)
        return page(
            "settings.html",
            request,
            claude=account.status(),
            form=form,
            groups=_category_groups(set(form.categories)),
            models=MODEL_CHOICES,
            deep_read_model=conf.deep_read_model if conf else cfg.DEEP_READ_MODEL,
            problem=problem,
            saved=request.query_params.get("saved") == "1",
            exists=config_path.exists(),
            config_path=str(config_path),
        )

    async def save_settings(request: Request) -> Response:
        if not await check_csrf(request):
            return Response("Invalid or missing form token; reload the page.", status_code=403)
        form = await SettingsForm.from_request(request)
        try:
            cfg.save_settings(
                config_path,
                categories=form.categories,
                interests=form.interests,
                values=form.values,
            )
        except cfg.ConfigError as exc:
            return page(
                "settings.html",
                request,
                422,
                claude=account.status(),
                form=form,
                groups=_category_groups(set(form.categories)),
                models=MODEL_CHOICES,
                deep_read_model=current_deep_read_model(),
                problem=str(exc),
                saved=False,
                exists=config_path.exists(),
                config_path=str(config_path),
            )
        return RedirectResponse("/settings?saved=1", status_code=303)

    app = Starlette(
        routes=[
            Route("/", index),
            Route("/interests", save_interests, methods=["POST"]),
            Route("/shortcut", shortcut, methods=["POST"]),
            Route("/digest/{digest_id:int}", show_digest),
            Route("/digest/{digest_id:int}/save/{kind}", save_digest),
            Route("/run", start_run, methods=["POST"]),
            Route("/weird", start_weird, methods=["POST"]),
            Route("/run", run_page, methods=["GET"]),
            Route("/run/status", run_status),
            Route("/health", health),
            Route("/heartbeat", heartbeat),
            Route("/icon.svg", icon),
            Route("/quit", quit_app, methods=["POST"]),
            Route("/background", background),
            Route("/fonts/{name}", font),
            Route("/claude", claude_login_page, methods=["GET"]),
            Route("/claude/login", claude_login, methods=["POST"]),
            Route("/claude/status", claude_status),
            Route("/settings", settings_page, methods=["GET"]),
            Route("/settings", save_settings, methods=["POST"]),
        ],
        middleware=[
            Middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"]),
            Middleware(ActivityTracker, activity=activity),
        ],
    )
    app.state.runner = runner
    app.state.stopping = False
    app.state.activity = activity
    if idle_exit:
        watchdog = threading.Thread(
            target=_idle_watchdog, args=(app, activity, idle_exit, busy, stop_soon), daemon=True
        )
        watchdog.start()
    app.state.account = account
    app.state.csrf = csrf_token
    return app
