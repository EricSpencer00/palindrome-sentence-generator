from experiments.incumbent_586_live_center_scene_extension_20260923 import build_payload


def test_live_center_scene_extension_is_independently_exact_and_longer() -> None:
    payload = build_payload()
    candidate = payload["candidate"]

    assert candidate["letters"] == 620
    assert candidate["length_delta"] == 34
    assert candidate["audit"]["independent_two_pointer_exact"]
    assert candidate["audit"]["project_validator_exact"]
    assert candidate["audit"]["project_normalizer_agrees"]
    assert candidate["audit"]["sha_equal"]
    assert candidate["live_seam"]["left_tape"] == "norasawaratonamat"
    assert candidate["live_seam"]["right_obligation"] == "tamanotarawasaron"
    assert candidate["live_seam"]["left_tape"] == candidate["live_seam"]["right_obligation"][::-1]
    assert len(candidate["live_seam"]["trace"]) == 17
    assert candidate["live_seam"]["final_residual"] == ""
    assert candidate["live_seam"]["committed_character_contradictions"] == 0


def test_scene_is_new_content_but_not_mislabeled_as_reader_evidence() -> None:
    payload = build_payload()
    candidate = payload["candidate"]

    assert payload["novelty_preflight"]["prior_exact_phrase_collisions"] == 0
    assert payload["method_scope"]["general_algorithm_claim"] is False
    assert candidate["provenance"]["borrowed_or_catalogue_text"] is False
    assert candidate["provenance"]["whole_word_order_mirror"] is False
    assert candidate["repair_debt"]["inserted_scene_pair_is_a_proper_palindromic_span"] is True
    assert candidate["repair_debt"]["blinded_readers_run"] is False
    assert payload["next_reader_facing_test"]["current_candidate_ready"] is False
