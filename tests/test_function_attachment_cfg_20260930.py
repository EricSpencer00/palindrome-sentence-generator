from experiments.function_attachment_cfg_20260930 import audit, run


def test_controls_and_independent_audit():
    result = run()
    assert result["provenance"]["function_words_live"]
    assert result["provenance"]["attachment_scope_live"]
    assert not result["provenance"]["complete_sentence_enumeration"]
    for row in result["candidates"]:
        assert row["audit"]["exact"]
        assert row["audit"]["sha256"] == row["audit"]["reverse_sha256"]


def test_normalizer_is_letter_level():
    assert audit("A man, a plan!")["normalized"] == "amanaplan"
    assert audit("A man, a plan!")["exact"] is False
