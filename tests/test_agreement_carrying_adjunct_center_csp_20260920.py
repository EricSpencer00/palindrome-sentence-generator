from experiments.agreement_carrying_adjunct_center_csp_20260920 import run


def test_agreement_slot_csp_keeps_actual_controls_and_bounded_equation_frontier():
    result = run()

    assert result["status"] == "completed_no_exact_closure"
    assert result["target_range"] == [39, 70]
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["novelty_preflight"]["signature_collision"] is False
    assert result["stats"]["outer_states"] == result["stats"]["semantic_shell_specs"]
    assert result["stats"]["outer_equation_pruned"] > 0
    assert result["stats"]["inner_states"] > 0
    assert result["stats"]["rendered_controls"] >= 1
    assert result["stats"]["exact"] == 0
    assert result["stats"]["mechanically_admitted"] == 0

    controls = [row for row in result["rows"] if row["control"]]
    assert controls
    assert any(39 <= row["audit"]["letters"] <= 70 for row in controls)
    for row in controls:
        assert row["rendered"].strip().endswith(".")
        assert row["provenance"]["equations_solved_during_search"] is True
        assert row["provenance"]["catalogue_imported"] is False
        assert row["provenance"]["finished_tape_reversed"] is False
        assert row["provenance"]["word_order_mirror"] is False
        assert row["provenance"]["rlaif_used"] is False
        assert row["agreement"]["subject_slot_compatible"] is True
        assert row["audit"]["two_pointer_exact"] is False
        assert row["audit"]["sha_equal_under_reversal"] is False
        assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]


def test_slot_state_carries_number_into_temporal_and_locative_realizations():
    result = run()
    rows = result["rows"]
    assert any(
        row["slot"]["kind"] == "temporal"
        and row["slot"]["number"] == row["agreement"]["subject_number"]
        for row in rows
    )
    assert any(
        row["slot"]["kind"] == "locative"
        and row["slot"]["number"] == row["agreement"]["subject_number"]
        for row in rows
    )
    assert any(row["slot"]["number"] == "pl" for row in rows)
    assert all(row["semantic_roles"]["word_order"] == "subject-verb-object-adjunct" for row in rows)
