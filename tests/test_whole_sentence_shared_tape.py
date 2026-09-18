from experiments.whole_sentence_shared_tape import checks


def test_whole_sentence_check_requires_exactness_and_rejects_repeated_units():
    gate = checks(["step", "on", "no", "pets"])
    assert gate["exact_palindrome"]
    assert not gate["length_band"]
    assert not gate["not_word_order_symmetry"]


def test_whole_sentence_check_rejects_nonpalindrome_without_changing_letters():
    gate = checks(["this", "is", "not", "one"])
    assert not gate["exact_palindrome"]


def test_whole_sentence_final_gate_rejects_catalogue_family_surface():
    gate = checks("Marge lets Hara see Sarah's telegram.".lower().rstrip(".").split())
    assert not gate["not_catalogue_family_derivative"]
