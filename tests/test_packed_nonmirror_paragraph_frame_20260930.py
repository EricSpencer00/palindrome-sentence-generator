from experiments.packed_nonmirror_paragraph_frame_20260930 import run
from experiments.packed_seam_grammar_20260927 import SEED, norm
from llm_palindrome.validator import is_palindrome


def test_nonmirror_frame_has_no_catalogue_or_rlaif_shortcut():
    result = run()
    assert result["provenance"]["complete_sentence_enumeration"] is False
    assert result["provenance"]["reverse_phrase_catalogue"] is False
    assert result["provenance"]["per_candidate_rlaif"] is False
    assert "non-mirrored" in result["topology"]


def test_exact_rows_are_independently_verified_and_controls_are_not_claimed():
    result = run()
    assert result["positive_control"]["independent_validator_exact"]
    assert result["positive_control"]["rendered"] == SEED
    for row in result["candidates"]:
        assert row["audit"]["exact"]
        assert row["audit"]["independent_validator_exact"]
        assert is_palindrome(row["rendered"])
        assert norm(row["rendered"]) == row["audit"]["normalized"]


def test_intact_controls_are_ordinary_nonpalindromic_prose():
    result = run()
    assert all(not row["audit"]["exact"] for row in result["controls"])
    assert all(row["kind"] == "intact_ordinary_prose" for row in result["controls"])
