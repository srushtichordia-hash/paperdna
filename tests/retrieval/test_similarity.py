"""Tests for cosine similarity and ranking."""

import math

import pytest

from paperdna.retrieval.errors import InvalidVectorError
from paperdna.retrieval.similarity import cosine, rank, validate_vector


class TestValidateVector:
    def test_returns_float_tuple(self):
        assert validate_vector([1, 2.5, 3]) == (1.0, 2.5, 3.0)

    def test_empty_vector_rejected(self):
        with pytest.raises(InvalidVectorError):
            validate_vector([])

    def test_dimension_mismatch_rejected(self):
        with pytest.raises(InvalidVectorError):
            validate_vector([1.0, 2.0], dimension=3)

    def test_non_finite_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with pytest.raises(InvalidVectorError):
                validate_vector([1.0, bad])

    def test_zero_norm_rejected(self):
        with pytest.raises(InvalidVectorError):
            validate_vector([0.0, 0.0, 0.0])

    def test_non_numeric_rejected(self):
        with pytest.raises(InvalidVectorError):
            validate_vector([1.0, "two", 3.0])


class TestCosine:
    def test_identical_vectors_are_one(self):
        assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_zero(self):
        assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_are_negative_one(self):
        assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_dimension_mismatch_rejected(self):
        with pytest.raises(InvalidVectorError):
            cosine([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_zero_vector_rejected_not_nan(self):
        with pytest.raises(InvalidVectorError):
            cosine([0.0, 0.0], [1.0, 1.0])


class TestRank:
    VECTORS = [
        [1.0, 0.0, 0.0],  # index 0
        [0.0, 1.0, 0.0],  # index 1
        [0.7, 0.7, 0.0],  # index 2
    ]

    def test_orders_by_descending_score(self):
        ranked = rank([1.0, 0.0, 0.0], self.VECTORS)
        assert [index for index, _ in ranked] == [0, 2, 1]
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_exact_scores(self):
        ranked = rank([1.0, 0.0, 0.0], self.VECTORS)
        assert ranked[0][1] == pytest.approx(1.0)
        assert ranked[1][1] == pytest.approx(1 / math.sqrt(2))

    def test_ties_keep_input_order(self):
        vectors = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        ranked = rank([1.0, 0.0], vectors)
        assert [index for index, _ in ranked] == [0, 1, 2]

    def test_empty_corpus_gives_empty_ranking(self):
        assert rank([1.0, 0.0], []) == []

    def test_query_dimension_mismatch_rejected(self):
        with pytest.raises(InvalidVectorError):
            rank([1.0, 0.0], [[1.0, 0.0, 0.0]])

    def test_invalid_corpus_vector_rejected(self):
        with pytest.raises(InvalidVectorError):
            rank([1.0, 0.0], [[1.0, 0.0], [math.nan, 1.0]])

    def test_invalid_query_vector_rejected(self):
        with pytest.raises(InvalidVectorError):
            rank([0.0, 0.0], self.VECTORS)
