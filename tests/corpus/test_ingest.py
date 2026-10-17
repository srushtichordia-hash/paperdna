"""Tests for batch ingestion and the local JSONL fixture."""

from pathlib import Path

from paperdna.corpus import ingest_records, load_jsonl
from paperdna.corpus.schema import IngestionResult, PaperRecord, RejectedRecord

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_corpus.jsonl"


def get_result() -> IngestionResult:
    return ingest_records(load_jsonl(str(FIXTURE)))


class TestFixtureIngestion:
    def test_fixture_stats(self):
        result = get_result()
        assert result.stats == {
            "loaded": 9,
            "validated": 6,
            "rejected": 3,
            "unique": 3,
            "duplicates": 3,
        }

    def test_fixture_unique_papers(self):
        result = get_result()
        assert [
            (p.source, p.source_id) for p in result.papers
        ] == [
            ("arxiv", "2101.00001"),
            ("fixture", "p002"),
            ("fixture", "p003"),
        ]

    def test_fixture_normalizes_messy_whitespace(self):
        result = get_result()
        p002 = result.papers[1]
        assert p002.title == "Gradient Stabilization in Deep Networks"
        assert "\n" not in p002.abstract
        assert p002.abstract == "A study of gradient stabilization techniques."

    def test_fixture_duplicate_match_reasons(self):
        result = get_result()
        assert sorted(d.matched_on for d in result.duplicate_records) == [
            "doi",
            "source_id",
            "title_year_author",
        ]

    def test_fixture_rejections_preserve_payloads(self):
        result = get_result()
        reasons = [r.reason for r in result.rejected]
        assert any("abstract" in reason for reason in reasons)
        assert any("published" in reason for reason in reasons)
        assert any("not a dictionary" in reason for reason in reasons)
        # The malformed JSONL line is preserved as raw text.
        assert any(isinstance(r.raw, str) for r in result.rejected)
        # The dict payloads are the original objects.
        for rejected in result.rejected:
            if isinstance(rejected.raw, dict):
                assert "title" in rejected.raw or "source" in rejected.raw


class TestStatisticsInvariants:
    def test_invariants_hold_for_fixture(self):
        stats = get_result().stats
        assert stats["loaded"] == stats["validated"] + stats["rejected"]
        assert stats["validated"] == stats["unique"] + stats["duplicates"]

    def test_invariants_hold_for_synthetic_batch(self):
        batch = [
            {"source": "s", "source_id": "1", "title": "T1", "abstract": "A",
             "authors": ["X"]},
            {"source": "s", "source_id": "1", "title": "T1 again", "abstract": "B",
             "authors": ["X"]},
            {"source": "s", "source_id": "2", "title": "T2"},  # missing abstract
            None,  # malformed
            "a string",  # malformed
        ]
        stats = ingest_records(batch).stats
        assert stats == {
            "loaded": 5,
            "validated": 2,
            "rejected": 3,
            "unique": 1,
            "duplicates": 1,
        }
        assert stats["loaded"] == stats["validated"] + stats["rejected"]
        assert stats["validated"] == stats["unique"] + stats["duplicates"]


class TestDuplicateProvenance:
    def test_every_duplicate_points_to_a_retained_paper(self):
        result = get_result()
        retained = {(p.source, p.source_id) for p in result.papers}
        for dup in result.duplicate_records:
            assert (dup.retained_source, dup.retained_source_id) in retained

    def test_doi_duplicate_reports_both_sides(self):
        result = get_result()
        doi_dups = [d for d in result.duplicate_records if d.matched_on == "doi"]
        assert len(doi_dups) == 1
        dup = doi_dups[0]
        assert (dup.duplicate.source, dup.duplicate.source_id) == (
            "openalex",
            "W1234567",
        )
        assert (dup.retained_source, dup.retained_source_id) == ("arxiv", "2101.00001")


class TestRejectedProvenance:
    def test_rejected_records_keep_original_payload(self):
        payload = {"source": "s", "source_id": "1", "title": "T"}
        result = ingest_records([payload])
        assert len(result.rejected) == 1
        assert result.rejected[0].raw is payload
        assert result.rejected[0].reason.startswith("abstract:")


class TestMalformedBatchesNeverAbort:
    def test_batch_of_junk_does_not_raise(self):
        junk = [None, 42, "text", ["list"], {"empty": "dict"}, 3.14, [], {}]
        result = ingest_records(junk)
        assert result.stats["loaded"] == len(junk)
        assert result.stats["validated"] == 0
        assert len(result.rejected) == len(junk)

    def test_valid_record_survives_junk_batch(self):
        good = {
            "source": "s",
            "source_id": "1",
            "title": "T",
            "abstract": "A",
            "authors": ["X"],
        }
        result = ingest_records([None, good, "junk"])
        assert len(result.papers) == 1
        assert isinstance(result.papers[0], PaperRecord)
        assert result.stats["rejected"] == 2

    def test_empty_batch(self):
        result = ingest_records([])
        assert result.papers == ()
        assert result.rejected == ()
        assert result.duplicate_records == ()
        assert result.stats == {
            "loaded": 0,
            "validated": 0,
            "rejected": 0,
            "unique": 0,
            "duplicates": 0,
        }


class TestLoadJsonl:
    def test_load_jsonl_keeps_unparseable_lines_as_text(self, tmp_path):
        path = tmp_path / "corpus.jsonl"
        path.write_text(
            '{"source": "s", "source_id": "1", "title": "T", "abstract": "A",'
            ' "authors": ["X"]}\n'
            "\n"
            "{not valid json\n",
            encoding="utf-8",
        )
        records = load_jsonl(str(path))
        assert len(records) == 2
        assert isinstance(records[0], dict)
        assert records[1] == "{not valid json"
        # And the batch still processes: 1 valid, 1 rejected.
        result = ingest_records(records)
        assert result.stats == {
            "loaded": 2,
            "validated": 1,
            "rejected": 1,
            "unique": 1,
            "duplicates": 0,
        }
        assert isinstance(result.rejected[0], RejectedRecord)
