from experiments.natural_dialogue_open_cycle_20260922 import audit, run


def test_authored_dialogue_search_records_a_real_frontier_and_gates_reader_claims():
    data = run()
    assert data["novelty_preflight"]["status"] == "collision_with_complete_clause_atomic_lanes"
    assert data["novelty_preflight"]["fixed_tape"] is False
    assert data["frontier"]["reachable_states"] > 0
    assert data["frontier"]["first_unsupported_frontier"] is not None
    assert data["reader_gate"]["status"] == "closed"
    assert data["status"] == "rejected_as_complete_clause_atomic_probe"


def test_independent_letter_audit():
    assert audit("A man, a plan, a canal: Panama!")["two_pointer_exact"]
    assert not audit("The lantern marks the harbor.")["two_pointer_exact"]
