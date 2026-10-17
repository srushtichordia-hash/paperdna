"""Embedding provider interface and the optional real-model provider.

The core package never imports torch or sentence-transformers: the real
provider loads the model only when it is explicitly instantiated, and the
import happens inside ``__init__`` rather than at module import time.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from paperdna.retrieval.errors import EmbedderUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@runtime_checkable
class Embedder(Protocol):
    """Anything that can embed texts into fixed-length float vectors."""

    name: str
    dimension: int | None  # None if not known until first use

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in order."""
        ...


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers model.

    Installing the ``embed`` extra is required:
    ``pip install -e ".[embed]"``. The model is downloaded on first use
    (about 90 MB for the default model, cached by Hugging Face) — never at
    module import time.

    ``all-MiniLM-L6-v2`` is Apache-2.0 licensed and produces 384-dimensional
    vectors suitable for titles and abstracts.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.name = model_name or DEFAULT_MODEL
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedderUnavailableError(
                "sentence-transformers is not installed. Install the optional "
                'extra with: pip install -e ".[embed]"'
            ) from exc
        logger.info("Loading embedding model %r (downloads on first use)", self.name)
        self._model = SentenceTransformer(self.name)
        self.dimension: int | None = int(
            self._model.get_sentence_embedding_dimension()
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts)
        return [[float(x) for x in vector] for vector in vectors]
