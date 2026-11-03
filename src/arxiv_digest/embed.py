"""Local text embeddings for matching papers to interests by meaning.

fastembed runs small ONNX models on the CPU, so there is no PyTorch download and
no GPU needed. The model is downloaded once into the arxiv-digest cache folder
(fastembed's own default is a temporary folder that is wiped on reboot).
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Protocol

import numpy as np


class EmbeddingUnavailable(Exception):
    """The local model could not be loaded (e.g. offline before the first download)."""


class Embedder(Protocol):
    name: str

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """One L2-normalized row per text."""
        ...

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """One L2-normalized row per text."""
        ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _default_threads() -> int:
    # A few threads finish nearly as fast as all of them and use far less total
    # CPU time (so less battery), especially on CPUs with efficiency cores.
    return max(1, min(4, os.cpu_count() or 1))


def _quiet_downloads() -> None:
    """Keep Hugging Face Hub progress bars and notices out of the CLI output."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="huggingface_hub")


class FastEmbedder:
    def __init__(self, model_name: str, cache_dir: Path, threads: int | None = None) -> None:
        self.name = model_name
        self._cache_dir = cache_dir
        self._threads = threads or _default_threads()
        self._model = None

    def _load(self):
        if self._model is None:
            _quiet_downloads()
            try:
                from fastembed import TextEmbedding

                self._cache_dir.mkdir(parents=True, exist_ok=True)
                self._model = TextEmbedding(
                    model_name=self.name, cache_dir=str(self._cache_dir), threads=self._threads
                )
            except Exception as exc:  # download, unknown model or runtime problems
                raise EmbeddingUnavailable(f"could not load {self.name}: {exc}") from exc
        return self._model

    def is_downloaded(self) -> bool:
        """Best-effort check, used only to announce the one-time download.

        fastembed names cache folders after the repository it downloads from
        (e.g. models--Qdrant--all-MiniLM-L6-v2-onnx), which ends in the model name.
        """
        if not self._cache_dir.exists():
            return False
        short = self.name.rsplit("/", 1)[-1].lower()
        return any(short in entry.name.lower() for entry in self._cache_dir.iterdir())

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._load()
        return _normalize(np.vstack(list(model.passage_embed(texts))))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._load()
        return _normalize(np.vstack(list(model.query_embed(texts))))
