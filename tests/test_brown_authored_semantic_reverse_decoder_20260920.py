from experiments.brown_authored_semantic_reverse_decoder_20260920 import audit, load_words, run


def test_brown_domains_and_authored_frames_are_nonempty():
    domains = load_words()
    assert all(domains[k] for k in ("DET", "ADJ", "AGENT", "ACTION", "OBJECT", "PREP", "PLACE"))
    result = run(max_left=1200, per_role=8)
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["complete_left_frames"] == 1200
    assert result["stats"]["reverse_states"] > 0


def test_controls_use_independent_audit():
    result = run(max_left=300, per_role=6)
    for row in result["controls"]:
        assert audit(row["rendered"]) == row["audit"]
    for row in result["rendered_candidates"]:
        assert row["audit"]["exact"] is False
