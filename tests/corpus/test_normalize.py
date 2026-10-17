"""Tests for text, author, and DOI normalization."""

from paperdna.corpus.normalize import normalize_authors, normalize_doi, normalize_text


class TestNormalizeText:
    def test_collapses_spaces_tabs_and_newlines(self):
        assert normalize_text("  a \t b\nc\r\n d  ") == "a b c d"

    def test_strips_ends(self):
        assert normalize_text("   padded   ") == "padded"

    def test_preserves_single_spaces(self):
        assert normalize_text("already clean") == "already clean"

    def test_unicode_nfc_normalization(self):
        # "e" + combining acute accent -> precomposed "é".
        assert normalize_text("e\u0301lan") == "\u00e9lan"


class TestNormalizeAuthors:
    def test_comma_separated_string(self):
        assert normalize_authors("Ada Lovelace, Grace Hopper") == (
            "Ada Lovelace",
            "Grace Hopper",
        )

    def test_list_of_strings(self):
        assert normalize_authors(["Alan Turing", "  Kurt  Godel "]) == (
            "Alan Turing",
            "Kurt Godel",
        )

    def test_drops_blank_entries(self):
        assert normalize_authors(["Ada Lovelace", "", "   "]) == ("Ada Lovelace",)

    def test_all_blank_gives_empty_tuple(self):
        assert normalize_authors(["", "  "]) == ()

    def test_wrong_type_returns_none(self):
        assert normalize_authors(42) is None
        assert normalize_authors({"a": 1}) is None
        assert normalize_authors(None) is None

    def test_list_with_non_string_member_returns_none(self):
        assert normalize_authors(["Ada Lovelace", 42]) is None


class TestNormalizeDoi:
    def test_bare_doi_lowercased(self):
        assert normalize_doi("10.1234/ML.001") == "10.1234/ml.001"

    def test_doi_prefix_form(self):
        assert normalize_doi("doi:10.1234/ML.001") == "10.1234/ml.001"
        assert normalize_doi("DOI:10.1234/ML.001") == "10.1234/ml.001"

    def test_url_forms(self):
        assert normalize_doi("https://doi.org/10.1234/ML.001") == "10.1234/ml.001"
        assert normalize_doi("http://dx.doi.org/10.1234/ML.001") == "10.1234/ml.001"

    def test_trailing_period_stripped(self):
        assert normalize_doi("10.1234/ml.001.") == "10.1234/ml.001"

    def test_surrounding_whitespace_stripped(self):
        assert normalize_doi("  10.1234/ML.001  ") == "10.1234/ml.001"

    def test_forms_are_equivalent(self):
        canonical = normalize_doi("10.1234/abc")
        assert normalize_doi("https://doi.org/10.1234/ABC") == canonical
        assert normalize_doi("doi:10.1234/Abc") == canonical

    def test_invalid_dois_return_none(self):
        assert normalize_doi("not-a-doi") is None
        assert normalize_doi("10.1234") is None  # missing suffix
        assert normalize_doi("http://example.com/10.1234/abc") is None
        assert normalize_doi("") is None
