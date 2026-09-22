import json
from pathlib import Path

from experiments.semantic_role_matrix_perfect_temporal_20260920 import run


def test_matrix_perfect_temporal_run_has_complete_controls_and_audits():
    result = run()
    stats = result["stats"]
    assert stats["heldout_clause_paths"] == 24
    assert stats["paired_grammar_states"] == 576
    assert stats["rendered_controls"] == 24
    assert stats["longest_rendered_control_letters"] >= 150
    assert stats["mechanical_exact_candidates"] == 0
    assert stats["exact_clean_above_38"] == 0
    assert result["novelty_preflight"]["registry_inspected"] is True
    assert all("audit" in row and "provenance" in row for row in result["rendered_controls"])
    assert all(row["provenance"]["post_hoc_repair"] is False for row in result["rendered_controls"])
