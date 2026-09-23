from experiments.incumbent_568_locative_inversion_active_svo_20260923 import run


def test_locative_inversion_masks_stay_complementary_and_parent_is_pinned():
    result = run()
    assert result["parent"]["normalized_letters"] == 568
    assert result["parent"]["sha256"] == "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
    for attempt in result["attempts"]:
        probe = attempt.get("probe")
        assert probe is not None
        assert probe["exact_local_equation"] is False
        assert probe["shared_reflected_boundaries"] == []
        if "residual_directed_mask_repair" in attempt:
            assert attempt["residual_directed_mask_repair"]["shared_reflected_boundaries"] == []


def test_one_residual_directed_internal_boundary_repair_advances_cursor():
    result = run()
    attempt = result["attempts"][1]
    before = attempt["probe"]
    after = attempt["residual_directed_mask_repair"]
    assert before["matched_outer_characters"] == 2
    assert before["first_mismatch"] == {"offset": 2, "left": "d", "right_reversed": "a"}
    assert after["matched_outer_characters"] == 4
    assert after["left_tape"].startswith("atad")
    assert after["first_mismatch"] == {"offset": 4, "left": "a", "right_reversed": "d"}


def test_one_slot_repairs_preserve_the_masks_but_expose_offset_six_obstructions():
    result = run()
    attempt = result["attempts"][1]
    magma, llama = attempt["single_slot_repairs"]
    assert magma["substitution"] == "field -> magma"
    assert magma["probe"]["matched_outer_characters"] == 6
    assert magma["probe"]["first_mismatch"] == {
        "offset": 6, "left": "s", "right_reversed": "g"
    }
    assert magma["probe"]["shared_reflected_boundaries"] == []
    assert llama["substitution"] == "field -> llama"
    assert llama["probe"]["matched_outer_characters"] == 6
    assert llama["probe"]["first_mismatch"]["offset"] == 6
    assert llama["probe"]["shared_reflected_boundaries"] == []


def test_attempt_is_not_misreported_as_an_exact_or_readable_child():
    result = run()
    assert result["stats"]["exact_local_equations"] == 0
    assert result["stats"]["independently_exact_children"] == 0
    assert result["stats"]["prospective_child_letters"] == 592
    assert result["stats"]["rendered_clause_pairs"] == 5
    assert result["shortcut_checks"]["candidate_admitted"] is False
    assert result["shortcut_checks"]["reader_claim"] is False
