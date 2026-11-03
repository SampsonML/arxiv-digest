"""The paper record shared by every part of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Paper:
    id: str  # arXiv identifier without the version, e.g. "2611.01234"
    version: int
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    primary_category: str
    published: datetime  # when version 1 was submitted (UTC)
    updated: datetime  # when the latest version was submitted (UTC)
    comment: str = ""
    journal_ref: str = ""
    doi: str = ""
    extra: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.id}"

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.id}"

    @property
    def text(self) -> str:
        """Title and abstract, the text that is matched and embedded."""
        return f"{self.title}. {self.abstract}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "title": self.title,
            "abstract": self.abstract,
            "authors": list(self.authors),
            "categories": list(self.categories),
            "primary_category": self.primary_category,
            "published": self.published.isoformat(),
            "updated": self.updated.isoformat(),
            "comment": self.comment,
            "journal_ref": self.journal_ref,
            "doi": self.doi,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Paper:
        return cls(
            id=data["id"],
            version=int(data["version"]),
            title=data["title"],
            abstract=data["abstract"],
            authors=tuple(data["authors"]),
            categories=tuple(data["categories"]),
            primary_category=data["primary_category"],
            published=datetime.fromisoformat(data["published"]),
            updated=datetime.fromisoformat(data["updated"]),
            comment=data.get("comment", ""),
            journal_ref=data.get("journal_ref", ""),
            doi=data.get("doi", ""),
        )
