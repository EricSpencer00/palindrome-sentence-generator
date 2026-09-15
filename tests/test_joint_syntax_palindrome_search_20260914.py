from __future__ import annotations

from experiments.joint_syntax_palindrome_search_20260914 import (
    PLANS,
    run,
    syntax_complete,
    syntax_state_possible,
    vocab,
)
from llm_palindrome.admission import normalize_letters


SEED = "an aide rips nine memos some men inspire diana".split()


def test_live_syntax_tracks_centerout_edges_and_replays_seed_shape() -> None:
    assert syntax_complete(SEED)
    # Center-out starts beside the midpoint, so the live left edge is a
    # suffix of the eventual prefix and the live right edge is a prefix of the
    # eventual suffix.
    assert syntax_state_possible(("memos",), ())
    assert syntax_state_possible(("nine", "memos"), ("some", "men"))
    assert syntax_state_possible(tuple(SEED[:5]), tuple(SEED[5:]))


def test_typed_inventory_keeps_function_articles_but_rejects_content_self_palindromes() -> None:
    words = set(vocab())
    assert "a" in words
    assert "i" in words
    assert "level" not in words


def test_joint_run_declares_live_roles_and_human_readability_gate() -> None:
    result = run(seeds=0, beam=32, candidate_limit=32, max_steps=2)
    config = result["config"]
    assert config["syntax_is_live_state_constraint"] is True
    assert config["character_residual_is_live_state_constraint"] is True
    assert {"VT", "NOUN", "ADJ"}.issubset(config["typed_state_roles"])
    assert config["machine_readability_certification"] is False
    assert result["reader_facing_next_test"]


def test_every_joint_record_has_independent_exact_audit() -> None:
    result = run(seeds=8, beam=600, candidate_limit=500, max_steps=120)
    for row in result["records"]:
        tape = normalize_letters(row["rendered"])
        assert tape == tape[::-1]
        assert row["independent_normalized_letters"] == tape
        assert row["independent_exact_audit"] is True
        assert row["syntax_complete"] is True
        assert row["mechanically_eligible"] is all(row["mechanical_checks"].values())
        assert row["reader_status"].startswith("human-unreviewed")

