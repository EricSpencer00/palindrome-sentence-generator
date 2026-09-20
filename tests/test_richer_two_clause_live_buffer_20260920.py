from experiments.richer_two_clause_live_buffer_20260920 import clauses, controls, run


def test_richer_bank_surfaces_and_controls_are_complete():
    bank = clauses()
    assert len(bank) > 70000
    assert all(text.split()[0] in {"the", "Alice", "Diana", "Marie", "Nora", "Peter", "Simon", "Victor"}
               for text, _ in bank)
    cs = controls()
    assert len(cs) == len(set(cs)) == 20
    assert all(text.endswith(".") and text.startswith("the ") for text in cs)


def test_richer_run_has_independent_gate_and_no_false_exact():
    result = run(state_limit=1000)
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["states"] == 1001
    assert result["stats"]["exact_gt38"] == 0
