import pytest

from training.audit_reader_ratings import audit


def rating(item, grammar=3, meaning=3, rater="Eric"):
    return dict(item=item, rater=rater, grammar=grammar, meaning=meaning)


def test_exact_gate_keeps_controls_and_missing_text_out():
    ratings = dict(study="s", rows=[rating("a"), rating("b"), rating("c")])
    packet = dict(study="s", items=[
        dict(item="a", text="Was it a car or a cat I saw?"),
        dict(item="b", text="Was it a car or a dog I saw?"),
    ])
    result = audit(ratings, packet)
    assert result["summary"]["counts"] == dict(exact=1, non_palindromes=1, unresolved=1)
    assert result["exact"][0]["item"] == "a"
    bad = result["non_palindromes"][0]
    mismatch = bad["first_mismatch"]
    assert mismatch["left"] != mismatch["right"]
    assert result["unresolved"] == [ratings["rows"][2]]
    assert result["summary"]["exact_length_bands"]["1-49"]["both_3"] == 1


def test_associations_rubric_is_separate_from_literal_word():
    result = audit(dict(study="s", rows=[rating("a", meaning=1)]),
                   dict(study="s", items=[dict(item="a", text="Never odd or even.")]))
    assert result["summary"]["exact_only"]["meaning_1"] == 1
    assert result["summary"]["exact_literal_association"]["count"] == 0


def test_empty_letter_tape_is_not_exact():
    result = audit(dict(study="s", rows=[rating("a")]),
                   dict(study="s", items=[dict(item="a", text="123?!")]))
    assert len(result["non_palindromes"]) == 1


@pytest.mark.parametrize("rows", [
    [rating("a"), rating("a")], [rating("a", grammar=True)],
    [rating("a", meaning=4)], [rating("a", grammar=1.5)],
])
def test_rejects_duplicate_or_invalid_labels(rows):
    with pytest.raises(ValueError):
        audit(dict(study="s", rows=rows))


def test_packet_must_match_and_ids_must_be_unique():
    ratings = dict(study="s", rows=[rating("a")])
    with pytest.raises(ValueError, match="study"):
        audit(ratings, dict(study="other", items=[]))
    with pytest.raises(ValueError, match="duplicate packet"):
        audit(ratings, dict(study="s", items=[dict(item="a", text="eve")]*2))


def test_multiple_raters_preserve_independent_ratings():
    result = audit(dict(study="s", rows=[rating("a"), rating("a", rater="other")]))
    assert result["summary"]["raters"] == 2
    assert len(result["unresolved"]) == 2
