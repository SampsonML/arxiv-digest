"""Local state in one SQLite file.

seen        every paper ID ever processed (kept forever; tiny), so no paper
            appears in two digests
papers      full metadata of recent papers (pruned after `retention_days`),
            for searching recent papers later
embeddings  cached embedding vectors, keyed by paper and model
runs        one row per run, for `status` and for working out what is new
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from .models import Paper

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen (
    id TEXT PRIMARY KEY,
    published TEXT NOT NULL,
    seen_at TEXT NOT NULL
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
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"{path} was written by a newer arxiv-digest; please upgrade arxiv-digest"
            )
        with self.db:
            self.db.executescript(_SCHEMA)
            self.db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self.db.close()

    # -- seen papers --------------------------------------------------------

    def seen_ids(self, ids: Iterable[str]) -> set[str]:
        ids = list(ids)
        found: set[str] = set()
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            marks = ",".join("?" * len(chunk))
            rows = self.db.execute(f"SELECT id FROM seen WHERE id IN ({marks})", chunk)
            found.update(row[0] for row in rows)
        return found

    def newest_seen_published(self) -> datetime | None:
        """Submission time of the newest paper processed so far."""
        row = self.db.execute("SELECT MAX(published) FROM seen").fetchone()
        return datetime.fromisoformat(row[0]) if row and row[0] else None

    def mark_seen(self, papers: Sequence[Paper], when: datetime | None = None) -> None:
        stamp = _iso(when or datetime.now(UTC))
        with self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO seen (id, published, seen_at) VALUES (?, ?, ?)",
                [(p.id, _iso(p.published), stamp) for p in papers],
            )
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
