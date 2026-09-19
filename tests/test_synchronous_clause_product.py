from experiments.synchronous_clause_product_20260919 import n

def test_first_last_word_gate_is_not_character_consumption():
    # Same endpoint letters pass the old shallow gate, but the streams diverge
    # immediately inside the words; a true synchronous parser must reject it.
    left = "ab"
    right = "ba"
    assert left[0] == right[-1]
    assert n(left) != n(right)[::-1]
