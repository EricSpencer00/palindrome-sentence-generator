"""Regression coverage for the 622 outer-shell reverse braid."""

from experiments.incumbent_622_outer_shell_braid_20260922 import (
    LEFT_WINDOW,
    NEW_LEFT,
    NEW_RIGHT,
    PARENT_SHA256,
    build_payload,
    consume_reverse_braid,
)


def test_join_consumes_two_cursors_and_closes_residual() -> None:
    state = consume_reverse_braid(
        NEW_LEFT,
        NEW_RIGHT,
        LEFT_WINDOW[0],
        486,
        ["Mara sees Aidan.", "Leon stops Mara.", "Nadia spots Leon."],
    )
    assert state["status"] == "accepted"
    assert state["final_residual"] == ""
    assert state["committed_character_contradictions"] == 0
    assert state["left_cursor_after"] == LEFT_WINDOW[0] + len(state["left_emission"])
    assert state["right_reverse_cursor_after"] == 486 - len(state["right_reverse_obligation"])
    assert len(state["trace"]) == len(state["left_emission"])
    assert len(state["clause_boundaries"]) == 3


def test_braid_payload_is_exact_growth_and_preserves_parent() -> None:
    payload = build_payload()
    row = payload["rows"][0]
    assert payload["parent"]["sha256"] == PARENT_SHA256
    assert row["independent_audit"]["normalized_letters"] == 648
    assert row["independent_audit"]["two_pointer_exact"] is True
    assert row["independent_audit"]["sha_equal"] is True
    assert row["online_join"]["accepted_join"]["final_residual"] == ""
    assert row["online_join"]["accepted_join"]["committed_character_contradictions"] == 0
    assert row["flags"]["complete_event_grammar"] is True
    assert row["flags"]["no_fragments_or_gibberish"] is True
    assert row["flags"]["inserted_units_unique"] is True
    assert row["repetition_delta"]["Mara stops rats."] == {"before": 2, "after": 1}
    assert row["repetition_delta"]["Eh, but a star spots Aram."] == {"before": 2, "after": 1}
    assert row["strict_global_checks"]["human_certified"] is False
    assert row["strict_global_checks"]["reader_status"].startswith("pending human")
