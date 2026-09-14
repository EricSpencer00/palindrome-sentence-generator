from experiments.boundary_chain_authoring_20260914 import audit, parse_text


def test_audit_independently_rejects_short_exact_sentence():
    result = audit("Never odd or even.")
    assert not result["mechanically_eligible"]
    assert result["letters"] == 14


def test_audit_requires_exact_shared_and_independent_tapes():
    result = audit("A baker shares warm bread with a child.")
    assert not result["mechanically_eligible"]
    assert result["independent_exactness"]["direct_symmetric_position_comparison"] is False


def test_parser_accepts_only_text_field_content():
    assert parse_text('{"text":"A fresh sentence."}') == ("A fresh sentence.", None)
    assert parse_text('{"other":"x"}') == (None, "text_schema_error")
