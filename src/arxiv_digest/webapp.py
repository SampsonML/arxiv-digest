"""The local web app: browse digests, run a new one, and edit settings.

`arxiv-digest app` serves it on 127.0.0.1 only. Two protections matter for a
local app that can change files and start runs:
  - Host check: requests must be addressed to 127.0.0.1 or localhost, which
    blocks DNS-rebinding attacks from web pages open in the same browser;
  - CSRF token: every form carries a random per-session token, so another web
    site cannot submit forms to the app on your behalf.
"""

from __future__ import annotations

import secrets
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from . import __version__, paths
from . import categories as cats
from . import config as cfg
from .html_render import KATEX_HEAD, KATEX_SCRIPTS, digest_body, stylesheet
from .pipeline import RunOptions, RunResult, run_with_config
from .store import Store

MODEL_CHOICES = ["sonnet", "haiku", "opus"]
RunFunction = Callable[[cfg.Config, RunOptions, Callable[[str], None]], RunResult]


def _default_run(config: cfg.Config, options: RunOptions, log: Callable[[str], None]) -> RunResult:
    return run_with_config(config, options, log=log)


# ---------------------------------------------------------------------------
# Background runs


@dataclass
class Job:
    state: str = "idle"  # idle | running | done | error
    lines: list[str] = field(default_factory=list)
    digest_id: int | None = None
    message: str = ""
    started: datetime | None = None


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
            self.job = Job(state="running", started=datetime.now())
            job = self.job
        self.thread = threading.Thread(target=self._work, args=(job, config, options), daemon=True)
        self.thread.start()
        return True

    def _work(self, job: Job, config: cfg.Config, options: RunOptions) -> None:
        try:
            result = self._run(config, options, job.lines.append)
        except Exception as exc:  # shown in the page; the server keeps running
            job.lines.append("".join(traceback.format_exception_only(exc)).strip())
            job.message = str(exc) or exc.__class__.__name__
            job.state = "error"
            return
        job.lines.extend(f"Note: {note}" for note in result.notes)
        job.digest_id = result.digest_id
        if result.digest is None:
            job.message = "No new papers since the last run."
        else:
            job.message = (
                f"{len(result.new_papers)} new papers · {len(result.shortlisted)} shortlisted"
                f" · {len(result.picks)} picked"
            )
        job.state = "done"


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
                    "output_dir": paths.default_output_dir(),
                    "max_papers": 15,
                    "top_picks": 3,
                    "use_claude": True,
                    "claude_model": "sonnet",
                },
            )
        return cls(
            categories=list(conf.categories),
            interests=list(conf.interests),
            values={
                "output_dir": paths.display(conf.output_dir),
                "max_papers": conf.max_papers,
                "top_picks": conf.top_picks,
                "use_claude": conf.use_claude,
                "claude_model": conf.claude_model,
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
                "output_dir": str(form.get("output_dir", "")).strip() or paths.default_output_dir(),
                "max_papers": number("max_papers", 15),
                "top_picks": number("top_picks", 3),
                "use_claude": form.get("use_claude") == "on",
                "claude_model": str(form.get("claude_model", "sonnet")).strip() or "sonnet",
            },
        )


# ---------------------------------------------------------------------------
# The app


def create_app(
    config_path: Path | None = None,
    *,
    run: RunFunction = _default_run,
    token: str | None = None,
) -> Starlette:
    config_path = config_path or paths.config_path()
    csrf_token = token or secrets.token_urlsafe(32)
    runner = Runner(run)
    env = Environment(
        loader=PackageLoader("arxiv_digest", "templates"),
        autoescape=select_autoescape(["html"]),
    )
    app_css = resources.files("arxiv_digest").joinpath("assets/app.css").read_text("utf-8")

    def load_config() -> tuple[cfg.Config | None, str | None]:
        if not config_path.exists():
            return None, None
        try:
            return cfg.load(config_path), None
        except cfg.ConfigError as exc:
            return None, str(exc)

    def page(name: str, request: Request, status: int = 200, **context: Any) -> HTMLResponse:
        template = env.get_template(name)
        html = template.render(
            request=request,
            css=Markup(stylesheet() + "\n" + app_css),
            csrf=csrf_token,
            version=__version__,
            job=runner.job,
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
            }
            for row in rows
        ]
        return page(
            "index.html",
            request,
            status,
            conf=conf,
            problem=problem,
            digests=digests,
            interests=interests if interests is not None else (conf.interests if conf else []),
            interests_error=interests_error,
            interests_saved=request.query_params.get("saved") == "1",
            summary=interest_summary,
            empty_interest=EMPTY_INTEREST,
        )

    async def index(request: Request) -> Response:
        return render_index(request)

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
        )
        runner.start(conf, options)  # if one is already running, just show it
        return RedirectResponse("/run", status_code=303)

    async def run_page(request: Request) -> Response:
        return page("run.html", request)

    async def run_status(request: Request) -> Response:
        job = runner.job
        return JSONResponse(
            {
                "state": job.state,
                "lines": job.lines,
                "digest_id": job.digest_id,
                "message": job.message,
            }
        )

    async def settings_page(request: Request) -> Response:
        conf, problem = load_config()
        form = SettingsForm.from_config(conf)
        return page(
            "settings.html",
            request,
            form=form,
            groups=_category_groups(set(form.categories)),
            models=MODEL_CHOICES,
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
                form=form,
                groups=_category_groups(set(form.categories)),
                models=MODEL_CHOICES,
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
            Route("/digest/{digest_id:int}", show_digest),
            Route("/run", start_run, methods=["POST"]),
            Route("/run", run_page, methods=["GET"]),
            Route("/run/status", run_status),
            Route("/settings", settings_page, methods=["GET"]),
            Route("/settings", save_settings, methods=["POST"]),
        ],
        middleware=[
            Middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"]),
        ],
    )
    app.state.runner = runner
    app.state.csrf = csrf_token
    return app
