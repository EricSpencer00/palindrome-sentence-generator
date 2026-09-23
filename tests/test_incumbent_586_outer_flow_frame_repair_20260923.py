from experiments.incumbent_586_outer_flow_frame_repair_20260923 import build_payload


def test_outer_flow_frame_repair_is_exact_and_preserves_length() -> None:
    payload = build_payload()
    candidate = payload["candidate"]

    assert candidate["letters"] == 586
    assert payload["stats"]["length_delta"] == 0
    assert candidate["audit"]["independent_two_pointer_exact"]
    assert candidate["audit"]["project_validator_exact"]
    assert candidate["audit"]["project_normalizer_agrees"]
    assert candidate["audit"]["sha_equal"]
    assert candidate["live_seam"]["left_emission"] == "wolfspotsflow"
    assert candidate["live_seam"]["right_obligation"] == "wolfstopsflow"
    assert candidate["live_seam"]["left_emission"][::-1] == candidate["live_seam"]["right_obligation"]
    assert len(candidate["live_seam"]["trace"]) == 13
    assert candidate["live_seam"]["final_residual"] == ""
    assert candidate["live_seam"]["committed_character_contradictions"] == 0


def test_outer_flow_frame_is_novel_and_does_not_claim_readability() -> None:
    payload = build_payload()
    candidate = payload["candidate"]

    assert payload["novelty_preflight"]["prior_exact_clause_collisions"] == 0
    assert candidate["provenance"]["borrowed_or_catalogue_text"] is False
    assert candidate["repair_debt"]["remaining_A_tub_shell"] is True
    assert candidate["repair_debt"]["human_readability_certified"] is False
    assert candidate["repair"]["rendered_outer_frame"] == [
        "Wolf spots flow.",
        "Wolf stops flow now, Noel.",
    ]
