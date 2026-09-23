from experiments.audit_incumbent_568_wide_chain_collision_20260923 import build_payload


def test_wide_chain_is_exact_but_rejected_for_reused_event_clauses():
    result = build_payload()
    candidate = result["candidate"]
    assert candidate["letters"] == 612
    assert candidate["growth_over_parent"] == 44
    assert candidate["normalized_sha256"] == "84a0d4a725d57c1a23d95c624ddb77366b834683131b4bb6bb1566aa8cc53a8c"
    assert candidate["audit"]["independent_outside_in_exact"] is True
    assert candidate["audit"]["project_validator_exact"] is True
    assert candidate["audit"]["hashes_equal"] is True
    assert candidate["whole_token_sequence_mirror"] is False
    assert result["working_status"] == "exact_but_novelty_rejected_not_promoted"


def test_collision_audit_finds_the_existing_linked_event_bank():
    result = build_payload()
    hits = result["novelty_preflight"]["collision_paths_by_clause"]
    assert result["novelty_preflight"]["revision"] == "8c35aba2"
    assert "Aidan stops Mara" in hits
    assert any("grow_568_with_linked_event_pair" in path for path in hits["Aidan stops Mara"])
    assert "Mara stops Sara" in hits
    assert any("incumbent-666-linked-scene-lattice" in path for path in hits["Mara stops Sara"])
    assert "Aras spots Aram" in hits
