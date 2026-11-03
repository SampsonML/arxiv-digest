"""Which papers a run skips: shown ones always, considered ones only under the
same interests and categories; and the upgrade of older databases."""

import json
import sqlite3
from dataclasses import replace

from conftest import NOW, FakeClient, FakeEmbedder, make_paper

from arxiv_digest.config import Interest, selection_fingerprint
from arxiv_digest.pipeline import run_digest
from arxiv_digest.store import SCHEMA_VERSION, Store


def _papers():
    return [
        make_paper(
            "2611.00001", "Neural emulators speed up cosmological simulations", hours_ago=20
        ),
        make_paper("2611.00002", "Flow matching for galaxy images", hours_ago=21),
        make_paper("2611.00003", "Rectified flow samplers", hours_ago=22),
        make_paper("2611.00004", "Medieval poetry", hours_ago=23),
    ]


def _run(config, store, papers):
    return run_digest(
        config, store, FakeClient(papers), FakeEmbedder(), log=lambda m: None, now=NOW
    )


def test_fingerprint_follows_interests_and_categories(make_config):
    config = make_config()
    same = make_config()
    assert selection_fingerprint(config) == selection_fingerprint(same)
    reordered = make_config(interests=list(reversed(config.interests)))
    assert selection_fingerprint(config) == selection_fingerprint(reordered)
    assert selection_fingerprint(config) != selection_fingerprint(make_config(categories=["cs.LG"]))
    changed = [replace(config.interests[0], keywords=["flow matching"]), config.interests[1]]
    assert selection_fingerprint(config) != selection_fingerprint(make_config(interests=changed))
    # Claude's search terms are derived from the interest, so they don't count.
    expanded = [replace(config.interests[0], search_terms=["x"]), config.interests[1]]
    assert selection_fingerprint(config) == selection_fingerprint(make_config(interests=expanded))


def test_changed_interests_reconsider_papers_that_were_never_shown(tmp_path, make_config):
    store = Store(tmp_path / "db.sqlite3")
    papers = _papers()
    first = _run(make_config(), store, papers)  # emulators: picks 00001
    shown = {p.candidate.paper.id for p in first.picks}
    assert "2611.00001" in shown and "2611.00002" not in shown

    # Same interests again: nothing new.
    assert _run(make_config(), store, papers).new_papers == []

    # New interest: the papers considered but not shown come back; shown ones don't.
    flows = Interest(name="Flows", description="flow matching", keywords=["flow matching"])
    again = _run(make_config(interests=[flows]), store, papers)
    ids = {p.id for p in again.new_papers}
    assert "2611.00002" in ids and not (ids & shown)
    assert "2611.00002" in {p.candidate.paper.id for p in again.picks}

    # And with that interest unchanged, nothing new again.
    assert _run(make_config(interests=[flows]), store, papers).new_papers == []


def test_upgrade_from_schema_3_marks_digest_papers_as_shown(tmp_path):
    path = tmp_path / "old.sqlite3"
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE seen (id TEXT PRIMARY KEY, published TEXT NOT NULL, seen_at TEXT NOT NULL);
        CREATE TABLE papers (id TEXT PRIMARY KEY, published TEXT NOT NULL, data TEXT NOT NULL);
        CREATE TABLE digests (
            id INTEGER PRIMARY KEY AUTOINCREMENT, generated TEXT NOT NULL, picked INTEGER NOT NULL,
            new INTEGER NOT NULL, markdown_path TEXT, html_path TEXT, data TEXT NOT NULL);
        PRAGMA user_version = 3;
        """
    )
    for pid in ("2611.00001", "2611.00002"):
        db.execute("INSERT INTO seen VALUES (?, '2026-11-02T00:00:00+00:00', 'x')", (pid,))
    entry = {"paper": make_paper("2611.00001").to_dict(), "interest": "x"}
    db.execute(
        "INSERT INTO digests (generated, picked, new, data) VALUES ('2026-11-02', 1, 2, ?)",
        (json.dumps({"entries": [entry]}),),
    )
    db.commit()
    db.close()

    store = Store(path)
    assert store.db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    # Shown before: always skipped. Only considered: looked at again once.
    assert store.skip_ids(["2611.00001", "2611.00002"], "any") == {"2611.00001"}
    assert store.list_digests()[0]["kind"] == "digest"
