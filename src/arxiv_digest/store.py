"""Local state in one SQLite file.

seen        every paper ID ever considered for a digest (kept forever; tiny):
            `shown` marks papers that appeared in a digest, so none appears
            twice; `fingerprint` records the interests and categories it was
            last considered under, so changing them reconsiders papers that
            were never shown
papers      full metadata of recent papers (pruned after `retention_days`),
            for searching recent papers later
embeddings  cached embedding vectors, keyed by paper and model
runs        one row per run, for `status` and for working out what is new
digests     every digest produced, as data, for the HTML views and web app
deep_reads  Claude's full-text summaries, so a paper is never read twice
expansions  search terms Claude suggested for each interest
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from .digest import Digest
from .models import DeepRead, Paper

SCHEMA_VERSION = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen (
    id TEXT PRIMARY KEY,
    published TEXT NOT NULL,
    seen_at TEXT NOT NULL,
    shown INTEGER NOT NULL DEFAULT 0,
    fingerprint TEXT
);
CREATE INDEX IF NOT EXISTS seen_published ON seen (published);
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    published TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS papers_published ON papers (published);
CREATE TABLE IF NOT EXISTS embeddings (
    paper_id TEXT NOT NULL,
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (paper_id, model)
);
CREATE TABLE IF NOT EXISTS digests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generated TEXT NOT NULL,
    picked INTEGER NOT NULL,
    new INTEGER NOT NULL,
    markdown_path TEXT,
    html_path TEXT,
    data TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'digest'
);
CREATE TABLE IF NOT EXISTS deep_reads (
    paper_id TEXT NOT NULL,
    key TEXT NOT NULL,
    data TEXT NOT NULL,
    created TEXT NOT NULL,
    PRIMARY KEY (paper_id, key)
);
CREATE TABLE IF NOT EXISTS expansions (
    key TEXT PRIMARY KEY,
    terms TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started TEXT NOT NULL,
    finished TEXT,
    window_start TEXT,
    window_end TEXT,
    fetched INTEGER,
    new INTEGER,
    picked INTEGER,
    digest_path TEXT
);
"""


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # The web app and the CLI may use the database at the same time.
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"{path} was written by a newer arxiv-digest; please upgrade arxiv-digest"
            )
        with self.db:
            self.db.executescript(_SCHEMA)
            if 0 < version < 4:
                self._upgrade_to_4()
            self.db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _columns(self, table: str) -> set[str]:
        return {row[1] for row in self.db.execute(f"PRAGMA table_info({table})")}

    def _upgrade_to_4(self) -> None:
        """Add `shown`/`fingerprint` to seen and `kind` to digests.

        Papers in earlier digests count as shown. Papers that were only
        considered get no fingerprint, so the next run looks at them once more
        under the current interests.
        """
        seen = self._columns("seen")
        if "shown" not in seen:
            self.db.execute("ALTER TABLE seen ADD COLUMN shown INTEGER NOT NULL DEFAULT 0")
        if "fingerprint" not in seen:
            self.db.execute("ALTER TABLE seen ADD COLUMN fingerprint TEXT")
        if "kind" not in self._columns("digests"):
            self.db.execute("ALTER TABLE digests ADD COLUMN kind TEXT NOT NULL DEFAULT 'digest'")
        shown: set[str] = set()
        for (data,) in self.db.execute("SELECT data FROM digests"):
            for entry in json.loads(data).get("entries", []):
                shown.add(entry["paper"]["id"])
        self.db.executemany("UPDATE seen SET shown = 1 WHERE id = ?", [(i,) for i in shown])

    def close(self) -> None:
        self.db.close()

    # -- seen papers --------------------------------------------------------

    def skip_ids(self, ids: Iterable[str], fingerprint: str) -> set[str]:
        """Papers to leave out of a run: shown in a digest before, or already
        considered under the same interests and categories (`fingerprint`)."""
        ids = list(ids)
        found: set[str] = set()
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            marks = ",".join("?" * len(chunk))
            rows = self.db.execute(
                f"SELECT id FROM seen WHERE id IN ({marks}) AND (shown = 1 OR fingerprint = ?)",
                [*chunk, fingerprint],
            )
            found.update(row[0] for row in rows)
        return found

    def newest_seen_published(self) -> datetime | None:
        """Submission time of the newest paper processed so far."""
        row = self.db.execute("SELECT MAX(published) FROM seen").fetchone()
        return datetime.fromisoformat(row[0]) if row and row[0] else None

    def mark_seen(
        self,
        papers: Sequence[Paper],
        fingerprint: str | None = None,
        *,
        shown: Iterable[str] = (),
        when: datetime | None = None,
    ) -> None:
        """Record papers as considered under `fingerprint`, and `shown` ones as shown."""
        stamp = _iso(when or datetime.now(UTC))
        with self.db:
            self.db.executemany(
                "INSERT INTO seen (id, published, seen_at, fingerprint) VALUES (?, ?, ?, ?)"
                " ON CONFLICT (id) DO UPDATE SET"
                " seen_at = excluded.seen_at, fingerprint = excluded.fingerprint",
                [(p.id, _iso(p.published), stamp, fingerprint) for p in papers],
            )
            self.db.executemany("UPDATE seen SET shown = 1 WHERE id = ?", [(i,) for i in shown])
        self.put_papers(papers)

    def put_papers(self, papers: Sequence[Paper]) -> None:
        """Keep papers' metadata (for search and get_paper) without marking them seen."""
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO papers (id, published, data) VALUES (?, ?, ?)",
                [(p.id, _iso(p.published), json.dumps(p.to_dict())) for p in papers],
            )

    def count_seen(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM seen").fetchone()[0]

    def recent_papers(self, since: datetime) -> list[Paper]:
        rows = self.db.execute(
            "SELECT data FROM papers WHERE published >= ? ORDER BY published DESC",
            (_iso(since),),
        )
        return [Paper.from_dict(json.loads(row[0])) for row in rows]

    def get_paper(self, paper_id: str) -> Paper | None:
        row = self.db.execute("SELECT data FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return Paper.from_dict(json.loads(row[0])) if row else None

    # -- embeddings ---------------------------------------------------------

    def get_embeddings(self, ids: Iterable[str], model: str) -> dict[str, np.ndarray]:
        ids = list(ids)
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            marks = ",".join("?" * len(chunk))
            rows = self.db.execute(
                "SELECT paper_id, vector FROM embeddings"
                f" WHERE model = ? AND paper_id IN ({marks})",
                [model, *chunk],
            )
            for paper_id, blob in rows:
                found[paper_id] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_embeddings(self, model: str, vectors: dict[str, np.ndarray]) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO embeddings (paper_id, model, vector) VALUES (?, ?, ?)",
                [
                    (pid, model, np.asarray(vec, dtype=np.float32).tobytes())
                    for pid, vec in vectors.items()
                ],
            )

    # -- runs ---------------------------------------------------------------

    def start_run(self, window_start: datetime, window_end: datetime) -> int:
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO runs (started, window_start, window_end) VALUES (?, ?, ?)",
                (_iso(datetime.now(UTC)), _iso(window_start), _iso(window_end)),
            )
        return int(cursor.lastrowid)

    def finish_run(
        self, run_id: int, *, fetched: int, new: int, picked: int, digest_path: Path | None
    ) -> None:
        with self.db:
            self.db.execute(
                "UPDATE runs SET finished = ?, fetched = ?, new = ?, picked = ?, digest_path = ?"
                " WHERE id = ?",
                (
                    _iso(datetime.now(UTC)),
                    fetched,
                    new,
                    picked,
                    str(digest_path) if digest_path else None,
                    run_id,
                ),
            )

    def last_run(self) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM runs WHERE finished IS NOT NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()

    # -- digests ------------------------------------------------------------

    def save_digest(
        self, digest: Digest, markdown_path: Path | None, html_path: Path | None
    ) -> int:
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO digests (generated, picked, new, markdown_path, html_path, data, kind)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    digest.generated.isoformat(),
                    len(digest.entries),
                    digest.new_papers,
                    str(markdown_path) if markdown_path else None,
                    str(html_path) if html_path else None,
                    json.dumps(digest.to_dict()),
                    digest.kind,
                ),
            )
        return int(cursor.lastrowid)

    def list_digests(self, limit: int = 200) -> list[sqlite3.Row]:
        """Newest first: id, generated, picked, new, markdown_path, html_path, kind."""
        return self.db.execute(
            "SELECT id, generated, picked, new, markdown_path, html_path, kind FROM digests"
            " ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def get_digest(self, digest_id: int) -> tuple[Digest, sqlite3.Row] | None:
        row = self.db.execute("SELECT * FROM digests WHERE id = ?", (digest_id,)).fetchone()
        if row is None:
            return None
        return Digest.from_dict(json.loads(row["data"])), row

    # -- deep reads ---------------------------------------------------------

    def get_deep_read(self, paper_id: str, key: str) -> DeepRead | None:
        row = self.db.execute(
            "SELECT data FROM deep_reads WHERE paper_id = ? AND key = ?", (paper_id, key)
        ).fetchone()
        return DeepRead.from_dict(json.loads(row[0])) if row else None

    def put_deep_read(self, paper_id: str, key: str, read: DeepRead) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO deep_reads (paper_id, key, data, created)"
                " VALUES (?, ?, ?, ?)",
                (paper_id, key, json.dumps(read.to_dict()), _iso(datetime.now(UTC))),
            )

    # -- search terms from Claude --------------------------------------------

    def get_expansion(self, key: str) -> list[str] | None:
        row = self.db.execute("SELECT terms FROM expansions WHERE key = ?", (key,)).fetchone()
        return list(json.loads(row[0])) if row else None

    def put_expansion(self, key: str, terms: Sequence[str]) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO expansions (key, terms, created) VALUES (?, ?, ?)",
                (key, json.dumps(list(terms)), _iso(datetime.now(UTC))),
            )

    # -- housekeeping -------------------------------------------------------

    def prune(self, retention_days: float = 90) -> None:
        """Drop full metadata and vectors of old papers; their IDs stay in `seen`."""
        cutoff = _iso(datetime.now(UTC) - timedelta(days=retention_days))
        with self.db:
            self.db.execute(
                "DELETE FROM embeddings WHERE paper_id IN "
                "(SELECT id FROM papers WHERE published < ?)",
                (cutoff,),
            )
            self.db.execute("DELETE FROM papers WHERE published < ?", (cutoff,))
            # Vectors cached by dry runs for papers that were never kept.
            self.db.execute("DELETE FROM embeddings WHERE paper_id NOT IN (SELECT id FROM papers)")
