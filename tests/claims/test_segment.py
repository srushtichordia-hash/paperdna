"""Tests for sentence segmentation and offsets."""

import pytest

from paperdna.claims import segment_sentences


def assert_offsets(text: str) -> None:
    for span in segment_sentences(text):
        assert text[span.start : span.end] == span.sentence


class TestBasics:
    def test_splits_on_period(self):
        text = "First sentence here. Second sentence here."
        spans = segment_sentences(text)
        assert [s.sentence for s in spans] == [
            "First sentence here.",
            "Second sentence here.",
        ]

    def test_question_and_exclamation(self):
        text = "Is this true? We show that it is! Final note."
        spans = segment_sentences(text)
        assert len(spans) == 3
        assert spans[0].sentence == "Is this true?"

    def test_single_sentence(self):
        spans = segment_sentences("Only one sentence.")
        assert len(spans) == 1

    def test_offsets_are_exact(self):
        text = "First sentence here. Second one! And a third?"
        assert_offsets(text)

    def test_whitespace_inside_sentence_preserved_verbatim(self):
        text = "A sentence with  double  spacing. Another."
        spans = segment_sentences(text)
        assert spans[0].sentence == "A sentence with  double  spacing."
        assert_offsets(text)


class TestEdgeCases:
    @pytest.mark.parametrize("text", ["", "   ", "\n\t\n", "\n \n"])
    def test_empty_or_whitespace_gives_no_sentences(self, text):
        assert segment_sentences(text) == ()

    def test_no_terminal_punctuation_is_one_sentence(self):
        text = "no punctuation at all"
        spans = segment_sentences(text)
        assert len(spans) == 1
        assert spans[0].sentence == "no punctuation at all"

    def test_decimal_number_does_not_split(self):
        text = "Accuracy improved by 4.2% on average. We conclude it works."
        spans = segment_sentences(text)
        assert len(spans) == 2
        assert "4.2%" in spans[0].sentence

    def test_e_g_abbreviation_does_not_split(self):
        text = "Models are used (e.g., for translation). We find gains."
        spans = segment_sentences(text)
        assert len(spans) == 2
        assert spans[0].sentence.endswith("(e.g., for translation).")

    def test_et_al_abbreviation_does_not_split(self):
        text = "As shown by Smith et al. Prior work ignored this. We find otherwise."
        spans = segment_sentences(text)
        assert len(spans) == 3

    def test_figure_abbreviation_does_not_split(self):
        text = "We show gains on 3 of 4 benchmarks (see Fig. 2). No difference was found."
        spans = segment_sentences(text)
        assert len(spans) == 2
        assert "Fig. 2" in spans[0].sentence

    def test_single_letter_initial_does_not_split(self):
        text = "A. Newton reported the effect. We confirm it."
        spans = segment_sentences(text)
        assert len(spans) == 2

    def test_closing_quote_stays_with_sentence(self):
        text = 'The author wrote "stop." Then the data followed.'
        spans = segment_sentences(text)
        assert len(spans) == 2
        assert spans[0].sentence == 'The author wrote "stop."'
        assert_offsets(text)

    def test_ellipsis_splits_after(self):
        text = "The effect was unclear... We find that it persists."
        spans = segment_sentences(text)
        assert len(spans) == 2

    def test_single_newlines_do_not_split(self):
        text = "A sentence that wraps\nonto a second line here. Next sentence."
        spans = segment_sentences(text)
        assert len(spans) == 2

    def test_blank_line_paragraph_break_splits(self):
        text = "First paragraph sentence.\n\nSecond paragraph sentence."
        spans = segment_sentences(text)
        assert len(spans) == 2
        assert_offsets(text)

    def test_non_string_input_gives_empty_tuple(self):
        assert segment_sentences(None) == ()  # type: ignore[arg-type]
        assert segment_sentences(42) == ()  # type: ignore[arg-type]

    def test_unicode_text(self):
        text = "La méthode est rapide. Nous constatons des gains."
        spans = segment_sentences(text)
        assert len(spans) == 2
