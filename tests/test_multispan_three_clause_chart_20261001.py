from experiments.multispan_scene_boundary_search_20261001 import run, audit


def test_three_clause_chart_has_live_residuals_and_no_shortcut_closure():
    result = run()
    assert result["search"]["bilateral_three_clause_states"] > 0
    assert result["search"]["residual_frontiers"] == 1680
    assert result["search"]["exact_closures"] == 0
    assert result["provenance"]["finished_tape_reversal"] is False
    assert result["provenance"]["repeated_units"] is False
    assert result["best_frontier"]["combined_audit"]["validator_exact"] is False
    assert result["best_frontier"]["combined_audit"]["sha_equal"] is False


def test_independent_audit_rejects_rendered_frontier():
    rendered = "Near the river, Lena found a blue button. At dawn, Mira opened the garden gate."
    checked = audit(rendered)
    assert checked["two_pointer_exact"] is False
    assert checked["validator_exact"] is False
    assert checked["sha_equal"] is False
