"""Semantic embedding and retrieval.

The heavy model stack (PyTorch, sentence-transformers) is optional: install
it with ``pip install -e ".[embed]"``. Nothing in this package imports it
until a ``SentenceTransformerEmbedder`` is explicitly instantiated.
"""

from paperdna.retrieval.embedder import DEFAULT_MODEL, Embedder, SentenceTransformerEmbedder
from paperdna.retrieval.errors import (
    EmbedderUnavailableError,
    InvalidVectorError,
    RetrievalError,
)
from paperdna.retrieval.eval import precision_at_k, recall_at_k
from paperdna.retrieval.index import PaperIndex, SearchResult
from paperdna.retrieval.preprocess import record_to_text
from paperdna.retrieval.similarity import cosine, rank, validate_vector

__all__ = [
    "DEFAULT_MODEL",
    "Embedder",
    "EmbedderUnavailableError",
    "InvalidVectorError",
    "PaperIndex",
    "RetrievalError",
    "SearchResult",
    "SentenceTransformerEmbedder",
    "cosine",
    "precision_at_k",
    "rank",
    "recall_at_k",
    "record_to_text",
    "validate_vector",
]
