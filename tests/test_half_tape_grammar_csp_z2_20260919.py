from experiments.half_tape_grammar_csp_z2_20260919 import run


def test_z2_half_tape_csp_recovers_anchor_with_independent_audits():
    result = run()
    assert result["stats"]["exact"] == 1
    assert result["stats"]["longest_exact_letters"] == 38
    row = result["candidates"][0]
    assert row["rendered"] == "An aide rips nine memos; some men inspire Diana."
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["sha256_forward"] == row["audit"]["sha256_reverse"]
    assert row["mechanically_admitted"]
    assert row["provenance"]["finished_tape_reversed"] is False
    assert row["provenance"]["catalogue_imported"] is False
