from experiments.incumbent_568_asymmetric_five_four_seam_20260923 import (
    PAIRS,
    direct_cross_token_reversals,
    independent_pointer_audit,
    live_obligation_trace,
    row,
    source_seam,
)
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_fresh_probe_is_an_asymmetric_clean_pair():
    for pair in PAIRS:
        probe = row(pair)
        assert len(pair.left_words()) == 5
        assert len(pair.right_words()) == 4
        assert direct_cross_token_reversals(pair) == []
        assert probe["novelty_and_shortcut_gate"]["passed"] is True
        assert probe["exact_insertion_possible"] is False


def test_live_owners_and_independent_audit_agree_on_best_obstruction():
    pair = PAIRS[0]
    trace = live_obligation_trace(pair)
    audit = independent_pointer_audit(pair.left, pair.right)
    assert trace["matched_outer_characters"] == 1
    assert trace["first_mismatch"]["left"]["letter"] == "b"
    assert trace["first_mismatch"]["right"]["letter"] == "t"
    assert audit["first_mismatch_offset"] == 1
    assert audit["equation_exact"] is False


def test_selected_parent_cuts_are_partial_words_and_mutually_mirrored():
    parent = json.loads((ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json").read_text())
    seam = source_seam(parent["rows"][0]["rendered"])
    assert seam["parent_normalized_cuts"] == [[99, 99], [469, 469]]
    assert seam["left_owner"]["surface_cut"] == "del|ivers"
    assert seam["right_owner"]["surface_cut"] == "revi|led"


def test_recorded_repair_moves_off_the_infeasible_and_archived_seam():
    artifact = json.loads((ROOT / "runs/incumbent-568-asymmetric-five-four-seam-20260923.json").read_text())
    assert len(artifact["novelty_preflight"]["checked_surface_phrases"]) == 16
    assert artifact["seam_viability"]["standalone_clause_insertion"] is False
    assert artifact["next_action"]["seam"] == [[148, 163], [405, 420]]
    assert "5-word/4-word" in artifact["next_action"]["operator"]
