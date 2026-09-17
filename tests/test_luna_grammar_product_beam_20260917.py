import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("luna_grammar_product_beam_20260917", ROOT / "experiments/luna_grammar_product_beam_20260917.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

def test_two_sided_product_renders_scene_grounded_prose_and_live_debt():
    result = MODULE.run()
    assert result["status"] == "completed_no_exact_closure"
    assert result["search"]["simultaneous_sides"]
    assert result["search"]["rlaif_per_candidate"] is False
    assert result["candidates"]
    assert all(row["rendered"].endswith(".") for row in result["candidates"])
    assert all(row["choices"] and row["scene"] for row in result["candidates"])
    assert all(row["obligation_ledger"][-1]["remaining_pair_debt"] > 0 for row in result["candidates"])
    assert "theme" in result["search"]["semantic_slots"]
    assert any("theme" in row["semantic_role_states"] for row in result["candidates"])
    assert max(row["letters"] for row in result["candidates"]) > 35

def test_independent_audit_and_repair_are_recorded():
    result = json.loads(MODULE.OUT.read_text())
    assert all(row["independent_audit"]["algorithm"] == "independent_two_pointer" for row in result["candidates"])
    assert all(not row["independent_audit"]["exact"] for row in result["candidates"])
    assert all(not row["anti_shortcut"]["word_order_mirror"] for row in result["candidates"])
    assert result["next_repair"]["operator"]
    expected = hashlib.sha256((ROOT / "experiments/luna_grammar_product_beam_20260917.py").read_bytes()).hexdigest()
    assert result["provenance"]["generator_sha256"] == expected
