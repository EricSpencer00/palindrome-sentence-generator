from experiments.incumbent_620_repeated_shell_repair_20260923 import build_payload


def test_repeated_shell_repair_preserves_exact_620_letter_length() -> None:
    payload = build_payload()
    candidate = payload["candidate"]

    assert candidate["letters"] == 620
    assert candidate["length_delta"] == 0
    assert candidate["audit"]["independent_two_pointer_exact"]
    assert candidate["audit"]["project_validator_exact"]
    assert candidate["audit"]["project_normalizer_agrees"]
    assert candidate["audit"]["sha_equal"]
    assert candidate["live_seam"]["after_left"] == "noelstopsflow"
    assert candidate["live_seam"]["after_right"] == "wolfspotsleon"
    assert candidate["live_seam"]["after_left"] == candidate["live_seam"]["after_right"][::-1]
    assert len(candidate["live_seam"]["trace"]) == 13
    assert candidate["live_seam"]["final_residual"] == ""
    assert candidate["live_seam"]["committed_character_contradictions"] == 0


def test_word_order_mirror_control_is_rejected_without_claiming_readability() -> None:
    payload = build_payload()
    candidate = payload["candidate"]

    assert candidate["repair"]["duplicate_occurrences_before"] == 2
    assert candidate["repair"]["duplicate_occurrences_after"] == 1
    assert payload["novelty_preflight"]["prior_exact_phrase_collisions"] == 0
    assert payload["method_scope"]["general_algorithm_claim"] is False
    assert candidate["provenance"]["borrowed_or_catalogue_text"] is False
    assert candidate["provenance"]["whole_word_order_mirror"] is True
    assert candidate["repair_debt"]["admitted_to_working_frontier"] is False
    assert payload["status"] == "exact_same_length_control_rejected_for_word_order_symmetry"
    assert payload["evaluation"]["failure_signature"]
    assert candidate["repair_debt"]["surrounding_prose_human_readable_certified"] is False
    assert payload["next_reader_facing_test"]["current_candidate_ready"] is False
