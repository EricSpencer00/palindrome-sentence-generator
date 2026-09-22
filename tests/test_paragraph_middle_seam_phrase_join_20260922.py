from experiments.paragraph_middle_seam_phrase_join_20260922 import (
    _phrase_rows,
    _two_pointer,
    _typed_domains,
)


def test_authored_carrier_equation_makes_exact_paragraph():
    # X="red rum", Y="murder" satisfies reverse(Y) = s + X only when
    # the synthetic corpus includes the exact normalized equation.
    sentences = [["red", "rum"], ["murder", "s"]]
    rows, _stats = _phrase_rows(sentences, maximum_words=2,
                                maximum_rows=20)
    row = next(row for row in rows
               if row["x_phrase"] == "red rum"
               and row["y_phrase"] == "murder s")
    assert row["equation"]["holds"] is True
    assert row["independent_exact_audit"]["exact"] is True


def test_two_pointer_is_independent_and_fail_closed():
    assert _two_pointer("No lemon, no melon.")["exact"] is True
    assert _two_pointer("ordinary prose")["exact"] is False


def test_typed_domains_require_complete_nominals_and_plural_subjects():
    tagged = [[("as", "CS"), ("an", "AT"), ("era", "NN"),
               ("arenas", "NNS"), ("assert", "VB")]]
    x_phrases, y_phrases = _typed_domains(tagged, 2)
    assert ("an", "era") in x_phrases
    assert ("arenas",) in y_phrases
    assert ("as",) not in x_phrases
