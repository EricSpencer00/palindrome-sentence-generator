from experiments.incumbent_568_internal_event_resegmentation_20260923 import build_payload


def test_internal_event_resegmentations_are_exact_children_of_pinned_568():
    payload = build_payload()
    assert payload["parent"]["letters"] == 568
    assert payload["parent"]["sha256_normalized"] == "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
    assert payload["seam"]["normalized_spans"] == [[20, 48], [520, 548]]
    assert [row["letters"] for row in payload["rows"]] == [584, 586]
    for row in payload["rows"]:
        assert row["audit"]["independent_outside_in_exact"] is True
        assert row["audit"]["project_validator_exact"] is True
        assert row["audit"]["hashes_equal"] is True
        assert row["live_residual"]["final_residual"] == ""
        assert all(item["consumed"] for item in row["live_residual"]["cursor_trace"])
        assert row["edit"]["retained_middle_unchanged"] is True
        assert row["anti_shortcut_audit"]["whole_token_sequence_is_mirror"] is False
        assert row["construction_debt"]["human_readability_certified"] is False


def test_phrases_are_not_reused_from_the_pre_experiment_head():
    payload = build_payload()
    assert payload["novelty_preflight"]["revision"] == "961ee718"
    assert payload["novelty_preflight"]["status"] == "passed"
    assert payload["novelty_preflight"]["hits"] == []


def test_local_tape_closes_at_character_boundaries_not_whole_word_order():
    payload = build_payload()
    for row in payload["rows"]:
        seam = row["live_residual"]
        assert seam["equation"]["left"] == seam["equation"]["right_obligation"]
        assert seam["characters_consumed"] == row["edit"]["left_letters"]
        assert row["anti_shortcut_audit"]["whole_token_sequence_is_mirror"] is False


def test_exact_but_semantically_incompatible_closure_is_rejected_and_redirects_method():
    proposal = build_payload()["rejected_proposals"][0]
    assert proposal["letters"] == 576
    assert proposal["local_equation_exact"] is True
    assert proposal["independent_outside_in_exact"] is True
    assert proposal["project_validator_exact"] is True
    assert "drawer" in proposal["rejection_reason"]
    assert "widen the seam" in proposal["next_operator"]
