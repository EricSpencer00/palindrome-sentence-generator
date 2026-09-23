from experiments.incumbent_620_wide_locative_seam_expansion_20260923 import build_payload


def test_wide_locative_seam_expansion_is_exact_and_longer_than_parent() -> None:
    payload = build_payload()
    candidate = payload["candidate"]

    assert candidate["letters"] == 626
    assert candidate["length_delta"] == 6
    assert candidate["audit"]["independent_two_pointer_exact"]
    assert candidate["audit"]["project_validator_exact"]
    assert candidate["audit"]["project_normalizer_agrees"]
    assert candidate["audit"]["sha_equal"]
    assert candidate["live_seam"]["left_tape"] == "norastopsaramonamatleonsawarat"
    assert candidate["live_seam"]["right_obligation"] == "tarawasnoeltamanomaraspotsaron"
    assert candidate["live_seam"]["left_tape"] == candidate["live_seam"]["right_obligation"][::-1]
    assert len(candidate["live_seam"]["trace"]) == 30
    assert candidate["live_seam"]["final_residual"] == ""
    assert candidate["live_seam"]["committed_character_contradictions"] == 0


def test_wider_edit_removes_a_fragment_without_word_order_mirror() -> None:
    payload = build_payload()
    candidate = payload["candidate"]

    assert candidate["repair"]["a_tub_question_occurrences_before"] == 2
    assert candidate["repair"]["a_tub_question_occurrences_after"] == 1
    assert candidate["live_seam"]["boundary_aligned_word_order_mirror"] is False
    assert payload["novelty_preflight"]["prior_exact_phrase_collisions"] == 0
    assert candidate["provenance"]["borrowed_or_catalogue_text"] is False
    assert candidate["repair_debt"]["inserted_scene_pair_is_a_proper_palindromic_span"] is True
    assert candidate["repair_debt"]["blinded_readers_run"] is False
    assert payload["next_reader_facing_test"]["current_candidate_ready"] is False
