"""Tests for label-order resolution (models differ; IDs are never trusted)."""

import pytest

from paperdna.nli import DEFAULT_LABEL_ORDER, canonical_label, resolve_label_map
from paperdna.nli.errors import NliError


class TestCanonicalLabel:
    def test_plain_labels(self):
        assert canonical_label("entailment") == "entailment"
        assert canonical_label("CONTRADICTION") == "contradiction"
        assert canonical_label(" Neutral ") == "neutral"

    def test_scifact_synonyms(self):
        assert canonical_label("SUPPORTS") == "entailment"
        assert canonical_label("REFUTES") == "contradiction"
        assert canonical_label("NOT_ENOUGH_INFO") == "neutral"
        assert canonical_label("not enough info") == "neutral"
        assert canonical_label("NEI") == "neutral"

    def test_unknown_returns_none(self):
        assert canonical_label("maybe") is None
        assert canonical_label(42) is None
        assert canonical_label(None) is None


class TestResolveLabelMap:
    def test_default_model_order(self):
        # cross-encoder/nli-deberta-v3-base: contradiction, entailment, neutral
        config = {0: "contradiction", 1: "entailment", 2: "neutral"}
        assert resolve_label_map(config) == (
            "contradiction",
            "entailment",
            "neutral",
        )

    def test_shuffled_order_resolved_correctly(self):
        # SciFact-style models often use entailment, neutral, contradiction.
        config = {0: "entailment", 1: "neutral", 2: "contradiction"}
        assert resolve_label_map(config) == ("entailment", "neutral", "contradiction")

    def test_string_indices(self):
        config = {"0": "REFUTES", "1": "SUPPORTS", "2": "NOT_ENOUGH_INFO"}
        assert resolve_label_map(config) == (
            "contradiction",
            "entailment",
            "neutral",
        )

    def test_missing_config_falls_back_to_documented_default(self):
        assert resolve_label_map(None) == DEFAULT_LABEL_ORDER
        assert resolve_label_map({}) == DEFAULT_LABEL_ORDER

    def test_unknown_label_raises(self):
        config = {0: "contradiction", 1: "entailment", 2: "maybe"}
        with pytest.raises(NliError):
            resolve_label_map(config)

    def test_duplicate_canonical_label_raises(self):
        config = {0: "SUPPORTS", 1: "entailment", 2: "neutral"}
        with pytest.raises(NliError):
            resolve_label_map(config)

    def test_missing_index_raises(self):
        config = {0: "contradiction", 1: "entailment"}  # no index 2
        with pytest.raises(NliError):
            resolve_label_map(config)

    def test_extra_index_raises(self):
        config = {0: "contradiction", 1: "entailment", 2: "neutral", 3: "maybe"}
        with pytest.raises(NliError):
            resolve_label_map(config)

    def test_non_integer_index_raises(self):
        config = {"zero": "contradiction", 1: "entailment", 2: "neutral"}
        with pytest.raises(NliError):
            resolve_label_map(config)
