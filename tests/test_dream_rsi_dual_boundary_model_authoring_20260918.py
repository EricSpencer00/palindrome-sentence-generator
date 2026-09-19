from experiments.dream_rsi_dual_boundary_model_authoring_20260918 import pair_row, run


def test_dual_boundary_pair_is_independently_audited():
    row = pair_row("the archivist records a note", "the reader carries a note", "test")
    assert "audit" in row and "residual" in row
    assert row["provenance"]["seed_scaffold_in_output"] is False


def test_model_authored_run_keeps_reader_gate_closed():
    result = run(per_prompt=1)
    assert result["stats"]["paired_worlds"] > 0
    assert result["stats"]["exact"] == 0
    assert result["reader_gate"]["status"] == "closed"
