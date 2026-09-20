from experiments.brown_authored_coordination_reverse_decoder_20260920 import audit, run

def test_coordination_lane_is_complete_and_audited():
    x=run(max_pairs=300)
    assert x["novelty_preflight"]["status"] == "passed"
    assert x["stats"]["complete_left_frames"] == 300
    assert x["stats"]["reverse_states"] > 0
    for row in x["controls"]: assert audit(row["rendered"]) == row["audit"]
