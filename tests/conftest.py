from __future__ import annotations

import zlib
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from arxiv_digest.config import Config, Interest
from arxiv_digest.matching import normalize
from arxiv_digest.models import Paper

NOW = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)


def make_paper(
    pid: str,
    title: str = "A paper",
    abstract: str = "Some abstract.",
    authors: tuple[str, ...] = ("A. Author",),
    hours_ago: float = 24,
    categories: tuple[str, ...] = ("astro-ph.CO",),
) -> Paper:
    published = NOW - timedelta(hours=hours_ago)
    return Paper(
        id=pid,
        version=1,
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        primary_category=categories[0],
        published=published,
        updated=published,
    )


class FakeEmbedder:
    """Bag-of-words vectors: texts sharing words get high similarity. No downloads."""

    name = "fake-model"

    def __init__(self) -> None:
        self.passage_calls = 0

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        vec = np.zeros(128, dtype=np.float32)
        for word in normalize(text).split():
            if len(word) > 3:
                vec[zlib.crc32(word.encode()) % 128] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self.passage_calls += 1
        return np.vstack([self._vector(t) for t in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vector(t) for t in texts])


class FakeClient:
    """Stands in for ArxivClient, returning papers inside the requested window."""

    def __init__(self, papers: list[Paper]) -> None:
        self.papers = papers
        self.calls: list[tuple[datetime, datetime]] = []

    def search(self, categories, start, end, *, progress=None):
        self.calls.append((start, end))
        return [p for p in self.papers if start <= p.published <= end]

    def newest(self, categories):
        return max(self.papers, key=lambda p: p.published, default=None)


@pytest.fixture
def make_config(tmp_path):
    def factory(**overrides) -> Config:
        values = dict(
            path=tmp_path / "config.toml",
            categories=["astro-ph.CO", "cs.LG"],
            interests=[
                Interest(
                    name="Fast simulations",
                    description="neural emulators that speed up cosmological simulations",
                    keywords=["emulator", "N-body"],
                ),
                Interest(name="Followed", authors=["Vera Rubin"]),
            ],
            output_dir=tmp_path / "digests",
        )
        values.update(overrides)
        return Config(**values)

    return factory
