import json
from pathlib import Path

from experiments.full_sequence_grammar_product_20260917 import (
    CATALOGUE_FIXTURE,
    PATTERNS,
    exact_audit,
    run,
    search_pattern,
)


def test_catalogue_fixture_validates_but_is_quarantined():
    result = run()
    fixture = result["fixture"]
    assert fixture["status"] == "quarantined_catalogue_fixture"
    assert fixture["admitted"] is False
    assert fixture["audit"]["exact"] is True
    assert fixture["audit"]["letters"] == 51


def test_outer_slot_product_has_live_mismatch_pruning_and_no_novel_output():
    result = run()
    assert result["status"] == "completed_no_novel_exact_closure"
    assert result["novel_exact_candidates"] == []
    assert all(row["mismatch_edges"] > 0 for row in result["searches"].values())
    assert all(row["exact_paths"] == 0 for row in result["searches"].values())


def test_engine_replays_fixture_geometry_when_catalogue_is_explicitly_enabled():
    found = search_pattern(PATTERNS["chain"], state_budget=250_000,
                           catalogue_fixture=True)
    assert any(row["audit"]["exact"] and row["audit"]["letters"] == 51
               for row in found.paths)


def test_artifact_matches_method_signature():
    run()
    artifact = json.loads(Path("runs/full-sequence-grammar-product-20260917.json").read_text())
    assert artifact["signature"].startswith("seedless-full-sequence-slot-product")
    assert "live_character_edges" in artifact["anti_shortcut_policy"] or artifact["searches"]
    assert exact_audit(artifact["fixture"]["rendered"])["exact"]
