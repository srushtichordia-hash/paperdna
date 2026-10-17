"""Tests for the embedder interface.

The real sentence-transformers provider is never constructed in ordinary
tests: no downloads, no torch import.
"""

import importlib.util
import sys

import pytest

import paperdna.retrieval
from paperdna.retrieval import Embedder, SentenceTransformerEmbedder
from paperdna.retrieval.embedder import DEFAULT_MODEL
from paperdna.retrieval.errors import EmbedderUnavailableError
from tests.retrieval.conftest import FakeEmbedder


def test_importing_retrieval_does_not_import_heavy_deps():
    assert "sentence_transformers" not in sys.modules
    assert "torch" not in sys.modules


def test_fake_embedder_satisfies_protocol():
    assert isinstance(FakeEmbedder(), Embedder)


def test_default_model_is_documented_choice():
    assert DEFAULT_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is not None,
    reason="sentence-transformers is installed; download test not applicable",
)
def test_missing_dependency_raises_helpful_error():
    with pytest.raises(EmbedderUnavailableError) as excinfo:
        SentenceTransformerEmbedder()
    assert 'pip install -e ".[embed]"' in str(excinfo.value)
    assert "sentence-transformers is not installed" in str(excinfo.value)


@pytest.mark.slow
def test_real_model_smoke():
    """Requires the embed extra and a model download; skipped by default."""
    pytest.importorskip("sentence_transformers")
    embedder = SentenceTransformerEmbedder()
    vectors = embedder.embed(["attention mechanisms", "gradient stabilization"])
    assert len(vectors) == 2
    assert all(len(v) == 384 for v in vectors)
