from llm_palindrome.refrain import compose_refrain
from llm_palindrome.validator import is_palindrome


ROWS = [
    {"text": "rats live on no evil star"},
    {"text": "stressed was i ere i saw desserts"},
    {"text": "no evil shahs live on"},
]


def test_refrain_is_palindromic_at_sentence_and_character_scales():
    out = compose_refrain(ROWS, 1000, seed=2)
    assert is_palindrome(out["text"])
    assert out["sentences"] == list(reversed(out["sentences"]))
    assert all(is_palindrome(sentence) for sentence in out["sentences"])
    assert out["max_sentence_uses"] == 2
    assert out["sentence_count"] == 5


def test_refrain_never_exceeds_target_or_repeats_adjacent_sentences():
    out = compose_refrain(ROWS, 70, seed=1)
    assert out["letters"] <= 70
    assert all(a != b for a, b in zip(out["sentences"], out["sentences"][1:]))


def test_refrain_theme_filters_material_and_rejects_unknown_theme():
    import pytest
    rows = ROWS + [{"text": "a man a plan a canal panama"}]
    out = compose_refrain(rows, 1000, theme="dark")
    assert "canal" not in out["text"].lower()
    assert out["theme"] == "dark"
    with pytest.raises(ValueError, match="unknown theme"):
        compose_refrain(rows, 1000, theme="missing")
