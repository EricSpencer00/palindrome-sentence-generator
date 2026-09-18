from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.bidirectional_attested_span_mining import common_lexicon  # noqa: E402
from experiments.endpoint_aware_constituent_repair import (  # noqa: E402
    BEAM,
    MAX_LETTERS,
    PATCHES_PER_STATE,
    PLANS,
    independent_two_pointer,
    prefix_debt,
    run,
    solve_joint_slots,
)


def test_surface_form_inventory_restores_observed_regular_inflections() -> None:
    vocabulary = common_lexicon(3.0)
    # These forms pass the existing dictionary rule but were previously
    # unreachable because the constructor only iterated headwords.
    assert {"rats", "reads", "writes", "farmers"}.issubset(vocabulary)


def test_active_constituent_repair_searches_beyond_the_paper_long_floor() -> None:
    assert MAX_LETTERS >= 160


def test_endpoint_intersection_exposes_impossible_determiner() -> None:
    blocked = prefix_debt("the", "letter")
    assert blocked["compatible"] is False
    assert blocked["first_mismatch"] == 0
    viable = prefix_debt("we", "few")
    assert viable["compatible"] is True
    assert viable["residual_debt"] == "f"


def test_solver_carries_character_debt_across_word_boundaries() -> None:
    # The known short palindrome is used only as a solver fixture.  The
    # experiment's shared gate excludes it from any prospective output.
    closures, stats = solve_joint_slots((("no",), ("lemon",), ("no",), ("melon",)))
    assert ("no", "lemon", "no", "melon") in closures
    assert stats["states_visited"] > 0
    assert independent_two_pointer("No lemon, no melon.")["exact"] is True


def test_run_logs_every_plan_round_and_keeps_reader_gate_closed() -> None:
    result = run(state_cap=2_000)
    assert result["config"]["plans"] == len(PLANS) == 32
    assert result["config"]["rounds"] == 8
    assert result["config"]["beam"] == BEAM == 64
    assert result["config"]["patches_per_state"] == PATCHES_PER_STATE == 32
    assert all(len(row["repair_rounds"]) == 8 for row in result["plans"])
    first_round = result["plans"][0]["repair_rounds"][0]
    assert "seed_proposal" in first_round
    assert "endpoint_intersections" in first_round
    assert "exact_closures" in first_round
    assert "No reader package" in result["reader_facing_next_test"]
