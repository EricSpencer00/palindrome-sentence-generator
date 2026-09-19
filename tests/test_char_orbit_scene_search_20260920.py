from experiments.char_orbit_scene_search_20260920 import LEXICON, FRAMES, audit, complete_gate, run


def test_audit_is_independent_and_exactness_is_strict():
    row = audit("A man, a plan.")
    assert row["normalized"] == "amanaplan"
    assert row["two_pointer_exact"] is False
    assert row["sha256_forward"] != row["sha256_reverse"]


def test_complete_controls_are_actual_prose_and_fsm_search_is_bounded():
    result = run()
    assert result["novelty_preflight"]["status"] == "passed"
    expected = sum(__import__('math').prod(len(LEXICON[slot]) for slot in frame) for frame in FRAMES)
    assert result["stats"]["visited"] == expected
    assert all(all(row["complete_clause_gate"].values()) for row in result["complete_prose_controls"])
    assert result["failure_and_next_discriminator"]["rlaif"] == "not used"


def test_candidate_provenance_bans_shortcuts():
    result = run()
    for row in result["candidates"]:
        assert row["provenance"]["finished_tape_reversed"] is False
        assert row["provenance"]["word_order_symmetry"] is False
        assert row["provenance"]["known_or_catalogue_palindrome"] is False
