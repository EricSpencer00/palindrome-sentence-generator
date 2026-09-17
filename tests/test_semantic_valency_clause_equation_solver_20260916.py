from experiments.semantic_valency_clause_equation_solver_20260916 import (
    CLAUSES, EXPERIMENT, novelty_preflight, run
)


def test_clause_equation_solver_expands_joint_states_and_reader_gate():
    result = run()
    assert result["experiment"] == EXPERIMENT
    assert result["novelty_preflight"]["passed"]
    assert result["outer_states_examined"] == len(CLAUSES[0]) * len(CLAUSES[2])
    assert result["states_expanded"] == 27
    assert result["exact_count"] == 0
    for row in result["best_rendered_candidates"]:
        assert row["rendered"]
        assert row["independent_exact_agreement"]
        assert not row["reader_gate"]["eligible"]
        assert row["anti_shortcut_flags"]["outer_equation_pruned"]
        assert row["next_repair"]


def test_solver_preflight_is_collision_free():
    result = novelty_preflight()
    assert result["passed"]
    assert result["exact_signature_collisions_before_render"] == []
