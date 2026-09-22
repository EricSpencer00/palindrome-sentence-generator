from experiments.packed_entity_attachment_grammar_20260930 import audit, bank_checks, run


def test_audit_independent_checks():
    result = audit("A man, a plan, a canal: Panama")
    assert result["two_pointer_exact"]
    assert result["sha_equal"]


def test_fresh_entity_bank_has_no_reverse_pairs():
    assert bank_checks()["reverse_pair_free"]


def test_result_has_provenance_and_reader_gate():
    result = run()
    assert result["reader_gate"].startswith("closed")
    assert result["provenance"]["independent_audits"]
