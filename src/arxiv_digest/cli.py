"""The `arxiv-digest` command."""

from __future__ import annotations

import os
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__, paths
from . import categories as cats
from . import config as cfg
from .arxiv_api import ArxivError
from .claude_cli import ClaudeCLI
from .pipeline import RunOptions, run_with_config
from .store import Store

app = typer.Typer(
    help="A daily digest of the new arXiv papers that match your interests.",
    no_args_is_help=True,
    add_completion=False,
)
categories_app = typer.Typer(help="Show or change the arXiv categories you watch.")
app.add_typer(categories_app, name="categories")

err = Console(stderr=True, soft_wrap=True)
out = Console(soft_wrap=True)

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config", "-c", help="Config file to use (default: ~/.config/arxiv-digest/config.toml)."
    ),
]


def _config_path(option: Path | None) -> Path:
    return option.expanduser() if option else paths.config_path()


def _fail(message: str) -> None:
    err.print(f"[red]Error:[/red] {escape(message)}")
    raise typer.Exit(1)


def _load(option: Path | None) -> cfg.Config:
    try:
        return cfg.load(_config_path(option))
    except cfg.ConfigError as exc:
        _fail(str(exc))
        raise  # unreachable; keeps type checkers happy


def _version(value: bool) -> None:
    if value:
        out.print(f"arxiv-digest {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version.")
    ] = False,
) -> None:
    pass


# ---------------------------------------------------------------------------
# init


def _split(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _ask_categories() -> list[str]:
    while True:
        answer = typer.prompt(
            "arXiv categories to watch, comma-separated (e.g. astro-ph.CO, cs.LG)"
        )
        codes, problems = [], []
        for code in _split(answer):
            official = cats.canonical(code)
            if official:
                codes.append(official)
            else:
                hint = cats.suggestions(code)
                problems.append(
                    f"{code!r}" + (f" (did you mean {', '.join(hint)}?)" if hint else "")
                )
        if problems:
            err.print(f"[yellow]Unknown:[/yellow] {escape(', '.join(problems))}")
            err.print("See all codes with `arxiv-digest categories list --all`.")
        elif codes:
            return codes


def _ask_interest() -> cfg.Interest:
    name = typer.prompt("Name of an interest (e.g. Weak lensing)").strip()
    while True:
        description = typer.prompt(
            "Describe it in a sentence or two (optional)", default="", show_default=False
        )
        keywords = _split(
            typer.prompt(
                "Keywords or phrases, comma-separated (optional)", default="", show_default=False
            )
        )
        authors = _split(
            typer.prompt(
                "Authors whose papers you always want, comma-separated (optional)",
                default="",
                show_default=False,
            )
        )
        if description.strip() or keywords or authors:
            return cfg.Interest(
                name=name, description=description.strip(), keywords=keywords, authors=authors
            )
        err.print("[yellow]Give at least a description, keywords or authors.[/yellow]")


@app.command()
def init(
    config: ConfigOption = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing config.")] = False,
) -> None:
    """Create a config file by answering a few questions."""
    path = _config_path(config)
    if path.exists() and not force:
        _fail(f"A config already exists at {path}. Edit it, or use --force to start over.")
    out.print(
        "Let's set up your arXiv digest. You can change everything later in the config file.\n"
    )
    categories = _ask_categories()
    output_dir = typer.prompt(
        "Folder for the daily Markdown files", default=paths.default_output_dir()
    )
    interests = [_ask_interest()]
    while typer.confirm("Add another interest?", default=False):
        interests.append(_ask_interest())
    text = cfg.render_template(categories, output_dir, interests)
    cfg.parse(text, path)  # sanity check before writing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    out.print(f"\n[green]Saved[/green] {escape(str(path))}")
    out.print("Next: [bold]arxiv-digest run[/bold] (the first run downloads a ~90 MB model once).")


# ---------------------------------------------------------------------------
# run


@app.command()
def run(
    config: ConfigOption = None,
    days: Annotated[
        float | None,
        typer.Option(help="Cover the last N days of submissions instead of 'new since last run'."),
    ] = None,
    include_seen: Annotated[
        bool,
        typer.Option(
            "--include-seen",
            help="Also include papers shown in earlier digests (e.g. after editing interests).",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Print the digest instead of saving it, and remember nothing."
        ),
    ] = False,
    no_model: Annotated[
        bool,
        typer.Option(
            "--no-model", help="Rank by keywords and authors only (skip the local model)."
        ),
    ] = False,
    no_claude: Annotated[
        bool,
        typer.Option("--no-claude", help="Choose papers on this computer only (no Claude)."),
    ] = False,
    no_deep_read: Annotated[
        bool,
        typer.Option("--no-deep-read", help="Skip Claude's full-text read of the top picks."),
    ] = False,
    no_open: Annotated[
        bool, typer.Option("--no-open", help="Don't open the HTML digest in the browser.")
    ] = False,
    show: Annotated[
        bool, typer.Option("--print", help="Also print the digest to the terminal.")
    ] = False,
) -> None:
    """Fetch new papers, rank them, write today's digest (Markdown and HTML) and open it."""
    conf = _load(config)
    options = RunOptions(
        days=days,
        include_seen=include_seen,
        dry_run=dry_run,
        use_embeddings=not no_model,
        use_claude=conf.use_claude and not no_claude,
        deep_read=not no_deep_read,
    )
    try:
        result = run_with_config(
            conf, options, log=lambda msg: err.print(f"[dim]{escape(msg)}[/dim]")
        )
    except ArxivError as exc:
        _fail(str(exc))

    for note in result.notes:
        err.print(f"[yellow]Note:[/yellow] {escape(note)}")
    if result.markdown is None:
        return
    if dry_run or show:
        out.print(result.markdown, markup=False, highlight=False)
    summary = (
        f"{len(result.new_papers)} new papers · {len(result.shortlisted)} shortlisted ·"
        f" {len(result.picks)} picked"
    )
    if result.path:
        err.print(f"[green]{summary}[/green]")
        err.print(f"  Markdown: {escape(str(result.path))}")
        err.print(f"  HTML:     {escape(str(result.html_path))}")
        if result.html_path and not no_open:
            webbrowser.open(result.html_path.as_uri())
    else:
        err.print(f"[green]{summary}[/green] (dry run: nothing saved)")


# ---------------------------------------------------------------------------
# categories


@categories_app.command("list")
def categories_list(
    config: ConfigOption = None,
    all_: Annotated[bool, typer.Option("--all", help="List every arXiv category.")] = False,
) -> None:
    """List the categories you watch (or all of arXiv's with --all)."""
    codes = sorted(cats.CATEGORIES) if all_ else _load(config).categories
    table = Table(show_header=False, box=None, pad_edge=False)
    for code in codes:
        table.add_row(f"[bold]{code}[/bold]", cats.CATEGORIES.get(code, "(not in arXiv's list)"))
    out.print(table)


@categories_app.command("add")
def categories_add(
    codes: Annotated[list[str], typer.Argument(help="Category codes, e.g. hep-th astro-ph.HE")],
    config: ConfigOption = None,
) -> None:
    """Start watching more categories."""
    try:
        conf, added = cfg.add_categories(_config_path(config), codes)
    except cfg.ConfigError as exc:
        _fail(str(exc))
    if added:
        out.print(f"Added {', '.join(added)}. Now watching: {', '.join(conf.categories)}")
    else:
        out.print(f"Nothing to add. Watching: {', '.join(conf.categories)}")


@categories_app.command("remove")
def categories_remove(
    codes: Annotated[list[str], typer.Argument(help="Category codes to stop watching")],
    config: ConfigOption = None,
) -> None:
    """Stop watching categories."""
    try:
        conf, removed = cfg.remove_categories(_config_path(config), codes)
    except cfg.ConfigError as exc:
        _fail(str(exc))
    if removed:
        out.print(f"Removed {', '.join(removed)}. Now watching: {', '.join(conf.categories)}")
    else:
        out.print(f"None of those were watched. Watching: {', '.join(conf.categories)}")


# ---------------------------------------------------------------------------
# web app


@app.command("app")
def web_app(
    config: ConfigOption = None,
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1 to serve the app on.")] = 8765,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Don't open the app in the browser.")
    ] = False,
    idle_exit: Annotated[
        int,
        typer.Option(
            "--idle-exit",
            help="Stop after this many seconds with no open app tab (0: run until stopped).",
        ),
    ] = 0,
) -> None:
    """Open the web app: browse digests, run a new one, and edit settings."""
    import threading

    import uvicorn

    from . import launcher
    from .webapp import create_app

    url = launcher.app_url(port)
    state = launcher.server_status(port)
    if state == "ours":
        out.print(f"arxiv-digest is already running at [bold]{url}[/bold]")
        if not no_browser:
            webbrowser.open(url)
        return
    if state == "other":
        _fail(f"Port {port} is used by another program. Try --port {port + 1}.")
    web = create_app(_config_path(config), idle_exit=idle_exit or None)
    server = uvicorn.Server(uvicorn.Config(web, host="127.0.0.1", port=port, log_level="warning"))

    def stop() -> None:  # the Quit button and the idle timer
        server.should_exit = True

    web.state.on_quit = stop
    out.print(f"arxiv-digest is running at [bold]{url}[/bold] (Quit in the app, or Ctrl+C)")
    if not no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    server.run()


@app.command("open")
def open_command(
    config: ConfigOption = None,
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1 to serve the app on.")] = 8765,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Start the app but don't open the browser.")
    ] = False,
) -> None:
    """Open the web app, starting it in the background if needed (what the launcher runs).

    A background app stops by itself 10 minutes after its last tab is closed.
    """
    from . import launcher

    explicit = config.expanduser() if config else None
    try:
        url, started = launcher.open_app(port, explicit, browser=not no_browser)
    except launcher.LauncherError as exc:
        _fail(str(exc))
    out.print(("Started" if started else "Already running") + f" at {url}")


shortcut_app = typer.Typer(help="Add or remove the arxiv-digest entry in your app launcher.")
app.add_typer(shortcut_app, name="shortcut")


@shortcut_app.command("install")
def shortcut_install(config: ConfigOption = None) -> None:
    """Add "arxiv-digest" (with its test-tube icon) to your system's app launcher."""
    from . import launcher

    explicit = config.expanduser().resolve() if config else None
    if explicit is None and os.environ.get("ARXIV_DIGEST_CONFIG"):
        explicit = paths.config_path().resolve()
    try:
        created = launcher.install_shortcut(explicit)
    except (launcher.LauncherError, OSError) as exc:
        _fail(str(exc))
    out.print("[green]Added arxiv-digest to your app launcher.[/green]")
    for path in created:
        out.print(f"  {escape(str(path))}")
    out.print(
        "It opens the app in your browser, starting it in the background when needed."
        f"\nIt runs: [dim]{escape(' '.join(launcher.launch_command(explicit)))}[/dim]"
    )


@shortcut_app.command("remove")
def shortcut_remove() -> None:
    """Remove the arxiv-digest entry from your app launcher."""
    from . import launcher

    removed = launcher.remove_shortcut()
    if removed:
        out.print("Removed " + ", ".join(escape(str(p)) for p in removed))
    else:
        out.print("No launcher entry to remove.")


# ---------------------------------------------------------------------------
# interests, status


@app.command()
def interests(config: ConfigOption = None) -> None:
    """Show the interests in your config."""
    conf = _load(config)
    for interest in conf.interests:
        out.print(f"[bold]{escape(interest.name)}[/bold]")
        if interest.description:
            out.print(f"  {escape(interest.description)}")
        for label, values in (
            ("keywords", interest.keywords),
            ("require", interest.require),
            ("exclude", interest.exclude),
            ("authors", interest.authors),
        ):
            if values:
                out.print(f"  [dim]{label}:[/dim] {escape(', '.join(values))}")
    out.print(f"\n[dim]Edit them in {escape(str(conf.path))}[/dim]")


@app.command()
def status(config: ConfigOption = None) -> None:
    """Show where files live and what the last run did."""
    path = _config_path(config)
    out.print(
        f"[bold]Config[/bold]     {escape(str(path))}" + ("" if path.exists() else "  (missing)")
    )
    if path.exists():
        try:
            conf = cfg.load(path)
            out.print(f"[bold]Digests[/bold]    {escape(str(conf.output_dir))}")
            out.print(f"[bold]Categories[/bold] {', '.join(conf.categories)}")
            out.print(
                f"[bold]Interests[/bold]  {escape(', '.join(i.name for i in conf.interests))}"
            )
            if conf.use_claude:
                claude = ClaudeCLI(conf.claude_command, conf.claude_model).status()
                colour = "green" if claude.logged_in else "yellow"
                out.print(
                    f"[bold]Claude[/bold]     [{colour}]{escape(claude.describe())}[/{colour}]"
                    f" · model {escape(conf.claude_model)}"
                )
                if claude.installed and claude.logged_in is False:
                    out.print("           Sign in with [bold]claude auth login[/bold]")
                elif not claude.installed:
                    out.print("           Install it from https://claude.com/claude-code")
            else:
                out.print("[bold]Claude[/bold]     off (use_claude = false)")
        except cfg.ConfigError as exc:
            out.print(f"[red]Config problem:[/red] {escape(str(exc))}")
    db = paths.database_path()
    out.print(f"[bold]Database[/bold]   {escape(str(db))}")
    if db.exists():
        store = Store(db)
        try:
            last = store.last_run()
            out.print(f"[bold]Seen[/bold]       {store.count_seen()} papers")
            if last:
                out.print(
                    f"[bold]Last run[/bold]   {last['finished'][:16].replace('T', ' ')} UTC ·"
                    f" {last['new']} new · {last['picked']} picked"
                    + (f" → {escape(last['digest_path'])}" if last["digest_path"] else "")
                )
        finally:
            store.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
