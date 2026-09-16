from __future__ import annotations

import pytest

from experiments.build_blinded_reader_package_20260916 import candidate_text, letters, shuffled, words


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
