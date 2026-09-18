from experiments.guided_word_mirror_authoring import derive_right, parse_left_phrases, screen_left


def test_right_half_is_derived_exactly_from_the_reversible_word_mapping():
    mapping = {"step": "pets", "on": "no", "no": "on", "pets": "step"}
    assert derive_right("step on", mapping) == "no pets"


def test_screen_preserves_an_exact_pair_even_if_it_fails_the_length_gate():
    mapping = {"step": "pets", "on": "no", "no": "on", "pets": "step"}
    row = screen_left("step on", mapping, existing_pairs=set(), novel_checker=lambda text: True)
    assert row["right"] == "no pets"
    assert row["checks"]["reverse_match"] is True
    assert row["checks"]["exact_palindrome"] is True
    assert "length_band" in row["rejection_codes"]


def test_left_phrase_response_requires_the_full_count():
    raw = '{"left_phrases":["one", "two", "three", "four"]}'
    phrases, error = parse_left_phrases(raw)
    assert error is None
    assert phrases[-1] == "four"
