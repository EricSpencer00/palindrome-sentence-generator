from experiments.qwen_phrase_bank_beam_20260920 import mechanical_gate


def test_qwen_gate_rejects_repeated_or_nested_exact_text():
    row = "Era oh will it puts a lot of. Nine post is an evening is to."
    gate = mechanical_gate(row)
    assert not gate["mechanically_admitted"]
    assert not gate["no_repeated_words"] or not gate["no_nested_word_span"]
