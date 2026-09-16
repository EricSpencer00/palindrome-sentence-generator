from experiments.online_grammar_state_slot_repair_20260916 import run, SIGNATURE, EXPERIMENT_ID

def test_heldout_repair_is_bounded_and_independently_audited():
    p = run()
    assert p["experiment_id"] == EXPERIMENT_ID
    assert p["signature"] == SIGNATURE
    assert p["stats"]["repair_trials"] == 3
    assert p["stats"]["exact"] == 0
    assert all(r["ledger_replay"]["replayed_after_replacement"] for r in p["rendered_candidates"])
    assert all(len(r["sha256"]) == 64 and not r["two_pointer"]["exact"] for r in p["rendered_candidates"])
