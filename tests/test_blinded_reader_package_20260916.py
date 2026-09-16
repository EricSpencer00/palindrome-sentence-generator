from __future__ import annotations

import pytest

from experiments.build_blinded_reader_package_20260916 import (
    _validated_provenance,
    _require_length_match,
    candidate_text,
    letters,
    shuffled,
    words,
)


def test_reader_controls_preserve_words_but_change_order() -> None:
    source = "A careful nurse records a dosage for the clinic."
    control = shuffled(source, seed=17)
    assert sorted(words(control)) == sorted(words(source))
    assert control != source


def test_candidate_text_rejects_short_or_nonpalindromic_rows() -> None:
    with pytest.raises(ValueError, match="exact palindrome"):
        candidate_text({"rendered": "A short sentence."})


def test_letters_removes_apostrophes_only_after_tokenization() -> None:
    assert letters("I can't stop") == "icantstop"


def test_provenance_must_be_structured_immutable_and_explicit() -> None:
    with pytest.raises(ValueError, match="structured record"):
        _validated_provenance({"provenance": "fresh"})
    with pytest.raises(ValueError, match="missing explicit flags"):
        _validated_provenance({"provenance": {}})
    valid = {
        key: False
        for key in (
            "source_sentences_copied",
            "catalogue_imported",
            "borrowed_text",
            "reversed_finished_sentence",
            "word_order_symmetry",
            "repeated_self_palindromic_unit",
        )
    }
    valid.update({"generator_sha256": "a" * 64, "source_sha256": "b" * 64})
    assert _validated_provenance({"provenance": valid}) == valid
    valid["catalogue_imported"] = True
    with pytest.raises(ValueError, match="disallowed source"):
        _validated_provenance({"provenance": valid})


def test_shuffle_rejects_palindrome_preserving_control() -> None:
    with pytest.raises(ValueError, match="non-palindromic shuffle"):
        shuffled("a a a a", seed=3)


def test_reader_control_length_is_bounded() -> None:
    _require_length_match("one two three", "one two", index=1, max_delta=6)
    with pytest.raises(ValueError, match="not length-matched"):
        _require_length_match("one two three four five", "one", index=2, max_delta=2)
