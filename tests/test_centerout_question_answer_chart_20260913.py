from experiments.centerout_question_answer_chart_20260913 import run


def test_second_chart_changes_topology_and_defers_subject():
    result = run(state_limit=100_000)
    assert result["config"]["different_topology_from_baker"]
    assert result["config"]["right_endpoint_first"]
    assert result["config"]["subject_deferred_until_endpoint_gate"]
    assert result["repair_operator"] == "endpoint_residual_gate"
    assert result["seed_control"]["rendered"] == "Can the patient tutor answer a clear question?"
    assert result["seed_control"]["independent_parse"]


def test_endpoint_gate_records_actual_rejection_and_no_candidate():
    result = run()
    frontier = result["deepest_live_frontier"]
    assert result["stats"]["endpoint_pairs"] == 9
    assert result["stats"]["endpoint_match_pairs"] == 0
    assert frontier["rejection"]["action"] == "endpoint_residual_gate"
    assert frontier["independent_replay"]["events_replayed"] == len(frontier["ledger"])
    assert result["exact_closures"] == []
    assert result["admitted_closures"] == []
    assert result["reader_status"].startswith("unreviewed")
