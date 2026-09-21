"""Small regression checks for the boundary-shifted reverse parser."""
import json
from pathlib import Path

from experiments.boundary_shifted_grammar_reverse_segmentation_20260920 import (
    boundary_transition_contract,
    build_clauses,
    independent_audit,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/boundary-shifted-grammar-reverse-segmentation-20260920.json"


def test_boundary_contract_closes_across_a_right_word_boundary():
    contract = boundary_transition_contract()
    assert contract["status"] == "closed"
    assert contract["matched_characters"] == 4
    assert contract["right_boundary_shifts"] == 1
    assert contract["boundary_events"][0]["from_word"] == "cba"
    assert contract["boundary_events"][0]["to_word"] == "d"


def test_banks_are_independent_complete_grammar_paths_and_run_is_audited():
    left = build_clauses("left")
    right = build_clauses("right")
    assert left and right
    left_content = {
        word.surface
        for clause in left
        for word in clause.words
        if word.surface not in {"a", "an", "the", "near", "under"}
    }
    right_content = {
        word.surface
        for clause in right
        for word in clause.words
        if word.surface not in {"a", "an", "the", "through", "beyond"}
    }
    assert left_content.isdisjoint(right_content)

    result = json.loads(RUN.read_text())
    assert result["stats"]["rendered_controls"] >= 12
    assert result["stats"]["mechanical_exact_candidates"] == 0
    assert result["boundary_transition_contract"]["status"] == "closed"
    assert all(
        independent_audit(row["rendered"])["sha256_forward"]
        == row["audit"]["sha256_forward"]
        for row in result["rendered_controls"]
    )
