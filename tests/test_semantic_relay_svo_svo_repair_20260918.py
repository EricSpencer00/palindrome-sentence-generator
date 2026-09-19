from experiments.semantic_relay_svo_svo_repair_20260918 import (
    RelayScene,
    direct_reversed_token_pairs,
    independent_audit,
    run,
    scene_row,
)


def test_reversed_token_pairs_are_a_hard_preflight_rejection():
    pairs = direct_reversed_token_pairs(("was", "noel", "saw", "leon"))
    assert {(pair["left"], pair["right"]) for pair in pairs} == {
        ("was", "saw"), ("noel", "leon")
    }


def test_independent_audit_agrees_on_known_small_palindrome():
    audit = independent_audit("A man, a plan, a canal: Panama.")
    assert audit["two_pointer_exact"] is True
    assert audit["sha_equal_under_reversal"] is True


def test_fresh_relay_requires_two_clause_semantic_link_and_live_frontier():
    scene = RelayScene("test", "archive", "reading_room", "archivist", "she",
                       "marks", "a", "ledger", "at", "dusk", "files", "one", "chart")
    row = scene_row(scene)
    assert row["semantic_relay"]["two_complete_finite_clauses"] is True
    assert row["semantic_relay"]["anaphoric_actor_link"] == {
        "actor": "archivist", "pronoun": "she"
    }
    assert row["semantic_relay"]["direct_reversed_token_pairs"] == []
    assert row["live_character_obligation"]["matched_outer_characters"] >= 1


def test_run_is_bounded_and_does_not_promote_non_exact_probes():
    result = run()
    assert result["config"]["scene_count"] == 8
    assert result["config"]["search"] == "fixed authored probes; no Cartesian lexical sweep"
    assert result["provenance"]["seed_or_polar_question_used"] is False
    assert result["stats"] == {
        "complete_clause_pairs": 8,
        "exact_closures": 0,
        "mechanically_admitted": 0,
        "reader_eligible": 0,
    }
