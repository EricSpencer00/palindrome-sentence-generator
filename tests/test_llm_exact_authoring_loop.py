from experiments.llm_exact_authoring_loop import checks, first_mismatch, parse_text


def test_parser_requires_a_text_field():
    assert parse_text('{"text":"Step on no pets."}') == ("Step on no pets.", None)
    assert parse_text('{"answer":"no"}')[1] == "text_schema_error"


def test_mismatch_reports_the_first_letter_that_breaks_exactness():
    assert first_mismatch("abca") == 1
    assert first_mismatch("abcba") is None


def test_whole_sentence_gate_rejects_short_known_and_repeated_shortcuts():
    gate = checks("Step on no pets.")
    assert gate["exact_palindrome"]
    assert not gate["length_band"]
    assert not gate["novel_catalogue"]
