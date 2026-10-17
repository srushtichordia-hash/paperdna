"""Tests for claim extraction: positives, negatives, hedges, attribution."""

import pytest

from paperdna.claims import CandidateClaim, extract_claims, extract_claims_batch
from paperdna.claims.cues import find_hedges
from tests.claims.conftest import make_record


class TestValidExtraction:
    def test_extracts_finding_sentence(self):
        record = make_record(abstract="We find that the method works.")
        result = extract_claims(record)
        assert len(result.claims) == 1
        claim = result.claims[0]
        assert claim.sentence == "We find that the method works."

    def test_claim_is_exact_source_text(self):
        abstract = "Background here. Our results indicate that accuracy improves."
        record = make_record(abstract=abstract)
        for claim in extract_claims(record).claims:
            assert abstract[claim.start : claim.end] == claim.sentence

    def test_source_attribution_preserved(self):
        record = make_record(source="openalex", source_id="W123")
        claim = extract_claims(record).claims[0]
        assert claim.source == "openalex"
        assert claim.source_id == "W123"

    def test_cue_is_verbatim_matched_text(self):
        record = make_record(abstract="We find that accuracy improves.")
        assert extract_claims(record).claims[0].cue.lower() == "we find"

    @pytest.mark.parametrize(
        "abstract",
        [
            "We find that the effect holds.",
            "We show that accuracy improves.",
            "We observe a significant gain.",
            "Our results suggest that the model generalizes.",
            "Our analysis indicates a strong effect.",
            "Results show improved robustness.",
            "We conclude that the effect is real.",
            "Error rates are associated with data size.",
            "We present a new benchmark.",
        ],
    )
    def test_finding_cues_selected(self, abstract):
        result = extract_claims(make_record(abstract=abstract))
        assert len(result.claims) == 1
        assert result.claims[0].sentence == abstract


class TestNegativeExamples:
    @pytest.mark.parametrize(
        "abstract",
        [
            "Previous studies relied on hand-crafted features.",
            "It is known that scaling helps.",
            "A similar approach has been proposed.",
            "Prior work ignored this effect.",
            "We trained the model on a public dataset.",
            "We then evaluated on three benchmarks.",
            "First, we collect the annotations.",
            "The implementation uses standard transformers.",
            "Deep learning has transformed computer vision.",
        ],
    )
    def test_non_claim_sentences_not_selected(self, abstract):
        assert extract_claims(make_record(abstract=abstract)).claims == ()


class TestUncertainty:
    def test_hedges_recorded(self):
        record = make_record(
            abstract="Our results suggest that the gains may diminish over time."
        )
        claim = extract_claims(record).claims[0]
        assert set(claim.hedges) == {"may", "suggest"}

    def test_claim_text_never_altered(self):
        abstract = "We find that the effect may be task dependent."
        claim = extract_claims(make_record(abstract=abstract)).claims[0]
        assert claim.sentence == abstract  # verbatim, hedges not stripped
        assert "may" in claim.hedges

    def test_is_associated_with_is_both_cue_and_hedge(self):
        record = make_record(
            abstract="Error rates are associated with data size."
        )
        claim = extract_claims(record).claims[0]
        assert "associated with" in claim.cue.lower()
        assert any("associated with" in h for h in claim.hedges)

    def test_non_claim_hedges_ignored(self):
        # "consistent with" is a hedge, but the sentence has no finding cue.
        record = make_record(
            abstract="These findings are consistent with prior scaling work."
        )
        assert extract_claims(record).claims == ()

    def test_hedges_deduplicated_and_positional(self):
        hedges = find_hedges("This may hold, and this may also hold.")
        assert hedges == ("may",)


class TestClassification:
    def test_finding_cue_beats_methods_cue(self):
        record = make_record(
            abstract="We find that the dataset improves accuracy."
        )
        result = extract_claims(record)
        assert len(result.claims) == 1
        assert result.stats["methods"] == 0

    def test_stats_invariants(self):
        abstract = (
            "Deep learning is popular. Previous studies used heuristics. "
            "We find that our method works. We trained on one dataset. "
            "Something entirely unclear here."
        )
        stats = extract_claims(make_record(abstract=abstract)).stats
        assert stats["sentences"] == 5
        assert (
            stats["sentences"]
            == stats["claims"] + stats["background"] + stats["methods"] + stats["other"]
        )
        assert stats["hedged"] <= stats["claims"]

    def test_sentence_order_preserved_in_claims(self):
        abstract = "We find one. Middle sentence. We show two."
        claims = extract_claims(make_record(abstract=abstract)).claims
        assert [c.sentence for c in claims] == ["We find one.", "We show two."]


class TestEdgeCases:
    def test_empty_abstract(self):
        result = extract_claims(make_record(abstract=""))
        assert result.claims == ()
        assert result.stats == {
            "sentences": 0,
            "claims": 0,
            "background": 0,
            "methods": 0,
            "other": 0,
            "hedged": 0,
        }

    def test_whitespace_abstract(self):
        assert extract_claims(make_record(abstract="   \n ")).claims == ()

    def test_abstract_without_claims(self):
        result = extract_claims(
            make_record(abstract="Background only. Prior work did this.")
        )
        assert result.claims == ()
        assert result.stats["sentences"] == 2

    def test_non_record_raises_type_error(self):
        with pytest.raises(TypeError):
            extract_claims({"abstract": "We find x."})  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            extract_claims(None)  # type: ignore[arg-type]

    def test_punctuation_heavy_abstract_does_not_crash(self):
        abstract = "Really?! We find that it does. (see Fig. 1; cf. prior.)"
        result = extract_claims(make_record(abstract=abstract))
        assert all(isinstance(c, CandidateClaim) for c in result.claims)

    def test_batch_preserves_order_and_isolation(self):
        records = [
            make_record(source_id="a", abstract="We find one."),
            make_record(source_id="b", abstract="Background only."),
            make_record(source_id="c", abstract="We show two."),
        ]
        results = extract_claims_batch(records)
        assert [r.stats["claims"] for r in results] == [1, 0, 1]
        assert results[0].claims[0].source_id == "a"
        assert results[2].claims[0].source_id == "c"
