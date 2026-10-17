"""Tests for record validation."""

import pytest

from paperdna.corpus.schema import PaperRecord, RejectedRecord
from paperdna.corpus.validation import validate_record


def valid_payload(**overrides) -> dict:
    payload = {
        "source": "arxiv",
        "source_id": "2101.00001",
        "title": "A Title",
        "abstract": "An abstract.",
        "authors": ["Ada Lovelace"],
    }
    payload.update(overrides)
    return payload


def get_record(payload) -> PaperRecord:
    outcome = validate_record(payload)
    assert isinstance(outcome, PaperRecord), f"unexpected rejection: {outcome}"
    return outcome


def get_rejection(payload) -> RejectedRecord:
    outcome = validate_record(payload)
    assert isinstance(outcome, RejectedRecord), f"unexpected record: {outcome}"
    return outcome


class TestValidRecords:
    def test_valid_record_round_trips(self):
        record = get_record(valid_payload())
        assert record.source == "arxiv"
        assert record.source_id == "2101.00001"
        assert record.title == "A Title"
        assert record.authors == ("Ada Lovelace",)

    def test_optional_fields_accepted(self):
        record = get_record(
            valid_payload(
                published="2021-01-05",
                url="https://arxiv.org/abs/2101.00001",
                doi="10.1234/ml.001",
            )
        )
        assert record.published == "2021-01-05"
        assert record.url == "https://arxiv.org/abs/2101.00001"
        assert record.doi == "10.1234/ml.001"

    def test_year_only_date_accepted(self):
        assert get_record(valid_payload(published="2020")).published == "2020"

    def test_whitespace_is_normalized(self):
        record = get_record(
            valid_payload(title="  Spaced \t Title ", abstract="Multi\nline  abstract.")
        )
        assert record.title == "Spaced Title"
        assert record.abstract == "Multi line abstract."

    def test_authors_as_comma_separated_string(self):
        record = get_record(valid_payload(authors="Ada Lovelace, Grace Hopper"))
        assert record.authors == ("Ada Lovelace", "Grace Hopper")

    def test_unknown_extra_keys_are_ignored(self):
        record = get_record(valid_payload(citation_count=12, venue="NeurIPS"))
        assert record.source == "arxiv"


class TestMissingFields:
    @pytest.mark.parametrize("field", ["source", "source_id", "title", "abstract"])
    def test_missing_required_field_is_rejected(self, field):
        payload = valid_payload()
        del payload[field]
        rejection = get_rejection(payload)
        assert rejection.reason.startswith(f"{field}: missing")

    def test_blank_title_is_rejected(self):
        rejection = get_rejection(valid_payload(title="   \n "))
        assert "title" in rejection.reason

    def test_missing_authors_is_rejected(self):
        payload = valid_payload()
        del payload["authors"]
        assert get_rejection(payload).reason.startswith("authors: missing")

    def test_empty_authors_list_is_rejected(self):
        rejection = get_rejection(valid_payload(authors=["", "  "]))
        assert "authors" in rejection.reason


class TestDates:
    @pytest.mark.parametrize("date", ["2023-02-30", "2023-13-01", "2023-00-10"])
    def test_invalid_calendar_dates_are_rejected(self, date):
        rejection = get_rejection(valid_payload(published=date))
        assert "published" in rejection.reason

    @pytest.mark.parametrize("date", ["2024-02-29", "2023-06-15", "2020"])
    def test_valid_dates_are_accepted(self, date):
        assert get_record(valid_payload(published=date)).published == date

    @pytest.mark.parametrize("date", ["22", "February 2023", "2023/06/15", "2023-6-5"])
    def test_malformed_date_strings_are_rejected(self, date):
        rejection = get_rejection(valid_payload(published=date))
        assert "published" in rejection.reason

    def test_non_string_date_is_rejected(self):
        rejection = get_rejection(valid_payload(published=2023))
        assert "published" in rejection.reason


class TestOptionalFieldTypes:
    def test_non_http_url_is_rejected(self):
        rejection = get_rejection(valid_payload(url="ftp://example.org/x"))
        assert "url" in rejection.reason

    def test_non_string_url_is_rejected(self):
        rejection = get_rejection(valid_payload(url=123))
        assert "url" in rejection.reason

    def test_missing_url_is_fine(self):
        assert get_record(valid_payload()).url is None

    def test_non_string_doi_is_rejected(self):
        rejection = get_rejection(valid_payload(doi=["10.1234/x"]))
        assert "doi" in rejection.reason

    def test_malformed_doi_is_dropped_but_record_valid(self):
        record = get_record(valid_payload(doi="not-a-doi"))
        assert record.doi is None

    def test_non_string_authors_is_rejected(self):
        rejection = get_rejection(valid_payload(authors=5))
        assert "authors" in rejection.reason

    def test_authors_list_with_non_string_member_is_rejected(self):
        rejection = get_rejection(valid_payload(authors=["Ada Lovelace", 42]))
        assert "authors" in rejection.reason


class TestMalformedInputs:
    @pytest.mark.parametrize("bad", [None, 42, "a string", ["a", "list"], 3.14])
    def test_non_dict_input_is_rejected_not_raised(self, bad):
        rejection = get_rejection(bad)
        assert rejection.reason == "record is not a dictionary"
        assert rejection.raw is bad

    def test_empty_dict_is_rejected(self):
        rejection = get_rejection({})
        assert rejection.reason.startswith("source:")

    def test_raw_payload_preserved_on_rejection(self):
        payload = valid_payload()
        del payload["abstract"]
        assert get_rejection(payload).raw is payload
