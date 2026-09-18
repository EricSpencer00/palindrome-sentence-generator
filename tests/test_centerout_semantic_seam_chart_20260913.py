from experiments.centerout_semantic_seam_chart_20260913 import run

def test_true_center_out_chart_reaches_six_matches():
    r=run(); assert r["config"]["starts_at_center_seam"]; assert r["config"]["expands_slot_indices_outward"]
    assert r["stats"]["matched_emissions"] >= 6
    assert r["deepest_live_frontier"]["independent_replay"]["events_replayed"] == len(r["deepest_live_frontier"]["ledger"])

def test_semantic_control_and_no_promotion():
    r=run(); assert r["seed_control"]["rendered"] == "The editor writes on only set ideas."
    assert r["seed_control"]["independent_parse"]; assert not r["seed_control"]["mechanically_admitted"]
    assert r["exact_closures"] == []; assert r["admitted_closures"] == []
