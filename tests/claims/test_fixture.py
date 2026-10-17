"""Tests against the hand-labeled sanity-check fixture."""

import pytest

from paperdna.claims import extract_claims, segment_sentences
from tests.claims.conftest import make_record


def records_from(labeled_abstracts):
    return [
        (
            item,
            make_record(
                source=item["source"],
                source_id=item["source_id"],
                abstract=item["abstract"],
            ),
        )
        for item in labeled_abstracts
    ]


def test_fixture_shape(labeled_abstracts):
    assert len(labeled_abstracts) >= 3
    for item in labeled_abstracts:
        assert item["abstract"].strip()
        assert item["sentences"], "every fixture abstract needs sentence labels"


def test_fixture_describes_itself_as_sanity_check(labeled_abstracts):
    # The _comment lives alongside the data; check via the file's content.
    from tests.claims.conftest import CLAIMS_FIXTURE

    text = CLAIMS_FIXTURE.read_text(encoding="utf-8")
    assert "sanity-check" in text


@pytest.mark.parametrize("index", range(4))
def test_fixture_labels_match_extraction(labeled_abstracts, index):
    item, record = records_from(labeled_abstracts)[index]
    expected = item["sentences"]

    spans = segment_sentences(record.abstract)
    assert len(spans) == len(expected), (
        f"{item['source_id']}: segmentation produced {len(spans)} sentences, "
        f"fixture labels {len(expected)}"
    )

    result = extract_claims(record)

    # Stats must match the positional labels.
    label_counts = {label["label"] for label in expected}
    assert label_counts <= {"claim", "background", "methods", "other"}
    assert result.stats["sentences"] == len(expected)
    assert result.stats["claims"] == sum(
        1 for label in expected if label["label"] == "claim"
    )
    assert result.stats["background"] == sum(
        1 for label in expected if label["label"] == "background"
    )
    assert result.stats["methods"] == sum(
        1 for label in expected if label["label"] == "methods"
    )

    # Every claim matches its labeled sentence positionally and textually.
    claim_positions = [i for i, label in enumerate(expected) if label["label"] == "claim"]
    assert len(result.claims) == len(claim_positions)
    for claim, position in zip(result.claims, claim_positions):
        label = expected[position]
        assert claim.sentence == spans[position].sentence
        assert record.abstract[claim.start : claim.end] == claim.sentence
        if "cue_contains" in label:
            assert label["cue_contains"].lower() in claim.cue.lower()
        if "hedges" in label:
            assert set(claim.hedges) == set(label["hedges"])


def test_fixture_attribution(labeled_abstracts):
    for item, record in records_from(labeled_abstracts):
        for claim in extract_claims(record).claims:
            assert claim.source == item["source"]
            assert claim.source_id == item["source_id"]
