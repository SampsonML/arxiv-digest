"""Scoring new papers against interests, and choosing the shortlist.

For each (paper, interest) pair:
  - a paper containing any `exclude` term is skipped for that interest;
  - a paper missing any `require` term is skipped for that interest;
  - similarity: cosine similarity between the paper's title + abstract and the
    interest's name, description and keywords (local embedding model);
  - z-score: how many standard deviations the similarity is above the average
    for that interest across the day's papers. Raw similarities sit in a narrow,
    model- and wording-dependent band (e.g. 0.60-0.80), so scores are compared
    on this relative scale instead; that also keeps a broadly worded interest
    from outranking a narrowly worded one;
  - keyword bonus: +1 per keyword found, at most +2 (in z-score units);
  - followed author: +100, which puts the paper above every topic match.

The shortlist keeps every followed-author paper plus the best papers for each
topic interest, so a busy topic cannot crowd out a quiet one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .config import Interest
from .matching import find_terms, match_authors, normalize
from .models import Paper

KEYWORD_BONUS = 1.0
MAX_KEYWORD_BONUS = 2.0
AUTHOR_BONUS = 100.0
MIN_PER_INTEREST = 5


@dataclass(frozen=True)
class Match:
    interest: str
    score: float
    similarity: float | None
    keywords: tuple[str, ...]
    authors: tuple[str, ...]
    zscore: float | None = None
    rank: int | None = None  # 1 = the day's closest paper to this interest
    of: int | None = None  # how many papers were ranked


@dataclass(frozen=True)
class Candidate:
    paper: Paper
    matches: tuple[Match, ...]  # best first

    @property
    def best(self) -> Match:
        return self.matches[0]

    @property
    def score(self) -> float:
        return self.best.score

    @property
    def followed_authors(self) -> tuple[str, ...]:
        names: list[str] = []
        for match in self.matches:
            names.extend(a for a in match.authors if a not in names)
        return tuple(names)

    def match_for(self, interest: str) -> Match | None:
        return next((m for m in self.matches if m.interest == interest), None)


def _similarity_stats(
    papers: Sequence[Paper],
    paper_vectors: Mapping[str, np.ndarray],
    query: np.ndarray,
) -> dict[str, tuple[float, float, int]]:
    """paper id -> (similarity, z-score, rank) for one interest."""
    ids = [p.id for p in papers if p.id in paper_vectors]
    if not ids:
        return {}
    sims = np.array([float(np.dot(paper_vectors[i], query)) for i in ids])
    std = float(sims.std())
    z = (sims - sims.mean()) / std if std > 0 else np.zeros_like(sims)
    order = np.argsort(-sims, kind="stable")
    ranks = np.empty(len(ids), dtype=int)
    ranks[order] = np.arange(1, len(ids) + 1)
    return {
        pid: (float(s), float(zz), int(r))
        for pid, s, zz, r in zip(ids, sims, z, ranks, strict=True)
    }


def score_papers(
    papers: Sequence[Paper],
    interests: Sequence[Interest],
    paper_vectors: Mapping[str, np.ndarray] | None = None,
    query_vectors: Mapping[str, np.ndarray] | None = None,
) -> list[Candidate]:
    """Every paper that matches at least one interest, with its matches."""
    stats: dict[str, dict[str, tuple[float, float, int]]] = {}
    if paper_vectors and query_vectors:
        for interest in interests:
            query = query_vectors.get(interest.name)
            if query is not None:
                stats[interest.name] = _similarity_stats(papers, paper_vectors, query)

    candidates = []
    for paper in papers:
        text = normalize(paper.text)
        matches = []
        for interest in interests:
            if interest.exclude and find_terms(text, interest.exclude):
                continue
            if interest.require and len(find_terms(text, interest.require)) < len(interest.require):
                continue
            keywords = tuple(find_terms(text, interest.keywords))
            authors = tuple(match_authors(paper.authors, interest.authors))
            interest_stats = stats.get(interest.name, {})
            similarity = zscore = rank = of = None
            if paper.id in interest_stats:
                similarity, zscore, rank = interest_stats[paper.id]
                of = len(interest_stats)
            if not (keywords or authors or similarity is not None):
                continue
            score = (zscore or 0.0) + min(KEYWORD_BONUS * len(keywords), MAX_KEYWORD_BONUS)
            if authors:
                score += AUTHOR_BONUS
            matches.append(
                Match(interest.name, score, similarity, keywords, authors, zscore, rank, of)
            )
        if matches:
            matches.sort(key=lambda m: m.score, reverse=True)
            candidates.append(Candidate(paper, tuple(matches)))
    return candidates


def shortlist(
    candidates: Sequence[Candidate], interests: Sequence[Interest], size: int
) -> list[Candidate]:
    """The papers worth a closer look: followed authors plus the top of each topic."""
    topics = [i for i in interests if i.has_topic]
    per_topic = max(MIN_PER_INTEREST, math.ceil(size / max(1, len(topics))))
    chosen: dict[str, Candidate] = {}
    for candidate in candidates:
        if candidate.followed_authors:
            chosen[candidate.paper.id] = candidate
    for interest in topics:
        scored = [
            (match.score, candidate)
            for candidate in candidates
            if (match := candidate.match_for(interest.name)) is not None
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        for _, candidate in scored[:per_topic]:
            chosen.setdefault(candidate.paper.id, candidate)
    ranked = sorted(chosen.values(), key=lambda c: c.score, reverse=True)
    if len(ranked) <= size:
        return ranked
    by_author = [c for c in ranked if c.followed_authors]
    others = [c for c in ranked if not c.followed_authors]
    kept = by_author + others[: max(0, size - len(by_author))]
    return sorted(kept, key=lambda c: c.score, reverse=True)


def _qualifies(match: Match, min_zscore: float) -> bool:
    return bool(
        match.authors or match.keywords or (match.zscore is not None and match.zscore >= min_zscore)
    )


def select_locally(
    shortlisted: Sequence[Candidate], max_papers: int, min_zscore: float
) -> list[Candidate]:
    """The final picks when no Claude ranking is used.

    A paper is picked if it has a followed author, a keyword hit, or a
    similarity z-score of at least `min_zscore` for some interest.
    """
    picked = []
    for candidate in shortlisted:
        good = [m for m in candidate.matches if _qualifies(m, min_zscore)]
        if good:
            # Put the qualifying matches first, so the paper is filed under one of them.
            rest = [m for m in candidate.matches if m not in good]
            picked.append(Candidate(candidate.paper, tuple(good + rest)))
    return sorted(picked, key=lambda c: c.score, reverse=True)[:max_papers]
