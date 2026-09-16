from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.morphosemantic_product_delay_20260916 import (
    Side,
    audit,
    cancel,
    render,
    run,
)


def test_output_delay_cancels_only_against_opposite_emitter() -> None:
    # Left owns ``abc``; a left continuation extends it, while a right edge
    # consumes its prefix and leaves the correct owner on the remainder.
    assert cancel("abc", 0, "d", 0) == ("abcd", 0)
    assert cancel("abc", 0, "ab", 1) == ("c", 0)
    assert cancel("abc", 0, "abcd", 1) == ("d", 1)
    assert cancel("abc", 0, "ax", 1) is None


def test_right_edge_words_render_back_in_ordinary_order() -> None:
    left = Side(state="COORD", words=("keeper", "guides", "a", "map"))
    right = Side(state="COORD", words=("story", "a", "carries", "farmer"))
    assert render(left, right) == "Keeper guides a map; farmer carries a story."


def test_bounded_product_is_not_a_hidden_complete_clause_cross_product() -> None:
    result = run(max_states=2_000)
    assert result["config"]["complete_clause_products"] is False
    assert result["config"]["fixed_tape_read"] is False
    assert result["stats"]["states"] == 2_000
    assert result["stats"]["exact_closures"] == 0
    assert result["stats"]["reader_eligible"] == 0


def test_audit_remains_independent_of_product_state() -> None:
    row = audit("Keeper guides a map; farmer carries a story.")
    assert row["letters"] == len(row["tape"])
    assert row["exact"] is False
    assert row["admitted"] is False
