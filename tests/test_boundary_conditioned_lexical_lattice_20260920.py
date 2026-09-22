from experiments.boundary_conditioned_lexical_lattice_20260920 import audit, run


def test_boundary_lattice_records_complete_prose_and_independent_audits():
    result = run()
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["lattice_walks"] == 1000
    assert result["stats"]["fresh_exact_gt38"] == 0
    assert result["rendered_candidates"]
    for row in result["rendered_candidates"]:
        assert row["complete_prose"] is True
        assert len(row["audit"]["sha256_forward"]) == 64
        assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
        for flag in (
            "finished_tape_reversal",
            "post_hoc_repair",
            "catalogue_text",
            "mirrored_token_units",
            "repeated_units",
            "fragment",
        ):
            assert row["provenance"][flag] is False


def test_boundary_audit_is_exactly_letter_level():
    assert audit("A man, a plan, a canal: Panama!")["exact"] is True
