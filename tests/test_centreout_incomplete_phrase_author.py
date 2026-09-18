from experiments.centreout_incomplete_phrase_author import parse, screen


def test_parse_requires_the_full_fixed_proposal_count():
    raw = '{"candidates":[' + ','.join('"x"' for _ in range(12)) + ']}'
    rows, error = parse(raw)
    assert error is None
    assert rows == ["x"] * 12
    assert parse('{"candidates":["x"]}')[1] == "need_exactly_12_candidates"


def test_screen_keeps_exactness_separate_from_reader_status():
    row = screen("Satan, oscillate my metallic sonatas!",
                 known={"satanoscillatemymetallicsonatas"})
    assert row["exact_letter_palindrome"] is True
    assert row["local_catalogue_absent"] is False
    assert row["no_self_palindromic_word"] is True


def test_screen_rejects_repeated_and_self_palindromic_words():
    row = screen("Anna, Anna!", known=set())
    assert row["distinct_words"] is False
    assert row["no_self_palindromic_word"] is False
