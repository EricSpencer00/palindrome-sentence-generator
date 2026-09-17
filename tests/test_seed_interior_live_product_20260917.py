import json
from pathlib import Path

from experiments.seed_interior_live_product_20260917 import OUT, SEED_TAPE, run


def test_live_product_recovers_seed_control_without_posthoc_generation():
    result = run()
    control = result["cells"]["control"]
    assert control["exact_paths"] == 1
    row = control["rows"][0]
    assert row["provenance"]["seed_regression_control"]
    assert row["audit"]["exact"]
    assert row["audit"]["sha256"] == row["audit"]["reverse_sha256"]
    assert row["normalized_length"] == len(SEED_TAPE) == 38
    assert not result["novel_exact_candidates"]


def test_all_cells_report_live_states_and_dead_frontiers():
    result = run()
    assert set(result["cells"]) == {"control", "lexical_change", "grammar_change", "combined"}
    for cell in result["cells"].values():
        assert cell["states"] > 0
        assert cell["dead_frontiers"]
        for row in cell["rows"]:
            assert row["provenance"]["live_product_search"]
            assert row["provenance"]["word_boundaries_independent"]
            assert row["audit"]["two_pointer_exact"] == row["audit"]["exact"]
            assert row["audit"]["sha256"] == row["audit"]["reverse_sha256"]


def test_run_artifact_matches_live_product_method():
    result = json.loads(OUT.read_text())
    assert result["method"] == "live outside-in product over independent seed-interior grammar automata"
    assert "heldout_lexical_branch_at_first_dead_frontier" == result["next_repair"]["operator"]
    assert all("every pushed state" in item for item in result["invariants"][:1])
