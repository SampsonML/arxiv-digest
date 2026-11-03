"""One digest run, from fetching to writing the Markdown file.

Which papers count as new:
  - normal run: everything submitted since the newest paper seen before (minus a
    2-day margin for papers announced late, e.g. after moderation), at most
    `max_catchup_days` back; papers already seen are dropped;
  - first run (or --include-seen): arXiv's latest announcement;
  - `--days N`: the N days of submissions up to the newest available paper, so
    asking on a Monday morning still finds papers rather than an empty weekend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from .arxiv_api import ArxivClient, announcement_start
from .claude_cli import ClaudeAnswer
from .config import Config
from .digest import Digest, build_digest
from .embed import Embedder, EmbeddingUnavailable
from .html_render import standalone_page
from .models import Paper
from .rank import ClaudeError, Pick, Ranker, local_picks
from .render import render_markdown
from .shortlist import Candidate, score_papers, select_locally, shortlist
from .store import Store

MARGIN = timedelta(days=2)
EMBED_BATCH = 256

Log = Callable[[str], None]


@dataclass
class RunOptions:
    days: float | None = None  # cover the last N days of submissions
    include_seen: bool = False  # also show papers from earlier digests
    dry_run: bool = False  # print only: write no file and remember nothing
    use_embeddings: bool = True
    use_claude: bool = True


@dataclass
class RunResult:
    window_start: datetime
    window_end: datetime
    fetched: int = 0
    new_papers: list[Paper] = field(default_factory=list)
    shortlisted: list[Candidate] = field(default_factory=list)
    picks: list[Pick] = field(default_factory=list)
    claude: ClaudeAnswer | None = None
    digest: Digest | None = None
    markdown: str | None = None
    html: str | None = None
    path: Path | None = None  # the Markdown file
    html_path: Path | None = None
    digest_id: int | None = None
    notes: list[str] = field(default_factory=list)


def digest_paths(output_dir: Path, day: datetime) -> tuple[Path, Path]:
    """YYYY-MM-DD.md/.html, or YYYY-MM-DD-2.md/.html etc. if that day already has one."""
    stem = f"{day:%Y-%m-%d}"
    number = 1
    while True:
        name = stem if number == 1 else f"{stem}-{number}"
        md, html = output_dir / f"{name}.md", output_dir / f"{name}.html"
        if not md.exists() and not html.exists():
            return md, html
        number += 1


def _embed(
    embedder: Embedder,
    store: Store,
    papers: list[Paper],
    config: Config,
    log: Log,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    vectors = store.get_embeddings((p.id for p in papers), embedder.name)
    missing = [p for p in papers if p.id not in vectors]
    if missing:
        downloaded = getattr(embedder, "is_downloaded", lambda: True)()
        if not downloaded:
            log(f"Downloading the local model {embedder.name} (one time, ~90 MB)…")
        log(f"Embedding {len(missing)} papers on this computer…")
        fresh: dict[str, np.ndarray] = {}
        for start in range(0, len(missing), EMBED_BATCH):
            batch = missing[start : start + EMBED_BATCH]
            matrix = embedder.embed_passages([p.text for p in batch])
            fresh.update({p.id: row for p, row in zip(batch, matrix, strict=True)})
        store.put_embeddings(embedder.name, fresh)
        vectors.update(fresh)
    topics = [i for i in config.interests if i.has_topic]
    queries: dict[str, np.ndarray] = {}
    if topics:
        matrix = embedder.embed_queries([i.query_text for i in topics])
        queries = {i.name: row for i, row in zip(topics, matrix, strict=True)}
    return vectors, queries


def run_digest(
    config: Config,
    store: Store,
    client: ArxivClient,
    embedder: Embedder | None,
    options: RunOptions | None = None,
    *,
    ranker: Ranker | None = None,
    log: Log = print,
    now: datetime | None = None,
) -> RunResult:
    options = options or RunOptions()
    now = (now or datetime.now(UTC)).astimezone(UTC)
    newest_seen = store.newest_seen_published()
    notes: list[str] = []

    if options.days is not None or newest_seen is None or options.include_seen:
        # Anchor on the newest announced paper, not on today's date: nothing is
        # announced on Friday and Saturday nights or on holidays.
        newest = client.newest(config.categories)
        anchor = newest.published if newest else now
        if options.days is not None:
            start = anchor - timedelta(days=options.days)
        else:
            start = announcement_start(anchor)
    else:
        earliest = now - timedelta(days=config.max_catchup_days)
        start = max(newest_seen - MARGIN, earliest)
        if newest_seen < earliest:
            notes.append(
                f"The previous run was more than {config.max_catchup_days:g} days ago;"
                f" this digest covers only the last {config.max_catchup_days:g} days."
            )
    result = RunResult(window_start=start, window_end=now, notes=notes)

    log(f"Fetching new papers in {', '.join(config.categories)} from arXiv…")
    papers = client.search(
        config.categories, start, now, progress=lambda done, total: log(f"  {done}/{total}")
    )
    result.fetched = len(papers)

    seen = set() if options.include_seen else store.seen_ids(p.id for p in papers)
    new = [p for p in papers if p.id not in seen]
    result.new_papers = new
    if new:
        result.window_start = min(p.published for p in new)
        result.window_end = max(p.published for p in new)

    run_id = None if options.dry_run else store.start_run(start, now)
    if not new:
        log("No new papers since the last run.")
        if run_id is not None:
            store.finish_run(run_id, fetched=result.fetched, new=0, picked=0, digest_path=None)
        return result

    vectors = queries = None
    if embedder is not None and options.use_embeddings:
        try:
            vectors, queries = _embed(embedder, store, new, config, log)
        except EmbeddingUnavailable as exc:
            notes.append(f"The local model was unavailable ({exc}); ranked by keywords only.")
    if vectors is not None:
        ranked_by = (
            f"Ranked on this computer by similarity ({embedder.name}),"
            " keywords and followed authors."
        )
    else:
        ranked_by = "Ranked by keywords and followed authors only."

    candidates = score_papers(new, config.interests, vectors, queries)
    result.shortlisted = shortlist(candidates, config.interests, config.shortlist_size)
    result.picks = local_picks(
        select_locally(result.shortlisted, config.max_papers, config.min_zscore)
    )
    if ranker is not None and options.use_claude and result.shortlisted:
        if not ranker.available():
            notes.append(
                "Claude Code (`claude`) wasn't found, so papers were chosen on this computer"
                " only. Install Claude Code to get Claude's picks and explanations."
            )
        else:
            log(f"Asking {ranker.label} to choose from {len(result.shortlisted)} papers…")
            try:
                ranked = ranker.rank(config, result.shortlisted)
            except ClaudeError as exc:
                notes.append(f"Claude couldn't rank the papers ({exc}); showing the local ranking.")
            else:
                result.picks = ranked.picks
                result.claude = ranked.answer
                answer = ranked.answer
                model = answer.model if answer and answer.model else ranker.label
                if answer and answer.duration_s is not None:
                    log(f"  Claude answered in {answer.duration_s:.0f} s")
                ranked_by = (
                    f"Chosen and explained by Claude ({model}) from {len(result.shortlisted)}"
                    " papers shortlisted on this computer by similarity, keywords and"
                    " followed authors."
                )

    result.digest = build_digest(
        config=config,
        picks=result.picks,
        new_papers=len(new),
        shortlisted=len(result.shortlisted),
        window_start=result.window_start,
        window_end=result.window_end,
        ranked_by=ranked_by,
        generated=datetime.now().astimezone(),
        notes=notes,
    )
    result.markdown = render_markdown(result.digest)
    result.html = standalone_page(result.digest)

    if not options.dry_run:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        result.path, result.html_path = digest_paths(config.output_dir, result.digest.generated)
        result.path.write_text(result.markdown, encoding="utf-8")
        result.html_path.write_text(result.html, encoding="utf-8")
        result.digest_id = store.save_digest(result.digest, result.path, result.html_path)
        store.mark_seen(new)
        assert run_id is not None
        store.finish_run(
            run_id,
            fetched=result.fetched,
            new=len(new),
            picked=len(result.picks),
            digest_path=result.path,
        )
        store.prune()
    return result


def run_with_config(
    config: Config, options: RunOptions | None = None, *, log: Log = print
) -> RunResult:
    """Set up the real arXiv client, local model and Claude, then run (CLI and web app)."""
    from . import paths
    from .claude_cli import ClaudeCLI
    from .embed import FastEmbedder
    from .rank import ClaudeRanker

    options = options or RunOptions()
    store = Store(paths.database_path())
    client = ArxivClient()
    embedder = (
        FastEmbedder(config.embedding_model, paths.model_cache_dir())
        if options.use_embeddings
        else None
    )
    ranker = None
    if options.use_claude and config.use_claude:
        cli = ClaudeCLI(config.claude_command, config.claude_model, timeout=config.claude_timeout)
        ranker = ClaudeRanker(cli)
    try:
        return run_digest(config, store, client, embedder, options, ranker=ranker, log=log)
    finally:
        client.close()
        store.close()
