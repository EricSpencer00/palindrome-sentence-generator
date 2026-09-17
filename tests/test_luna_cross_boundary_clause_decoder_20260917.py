import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "luna_cross_boundary_clause_decoder_20260917",
    ROOT / "experiments/luna_cross_boundary_clause_decoder_20260917.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_novelty_preflight_and_typed_inventory():
    result = MODULE.novelty_preflight()
    assert result["status"] == "passed"
    assert "fixed tape" in result["rejected_shortcuts"]
    assert MODULE.complete_clauses()


def test_obligation_trie_is_character_level_and_accepts_reverse():
    trie = MODULE.CharacterObligationTrie("a bell")
    assert trie.accepts("lleba")
    seams = trie.boundary_obligations("a bell")
    assert seams[0]["character_level"]
    assert seams[0]["seam"] == "ab"


def test_decoder_realizes_two_complete_svo_pp_clauses_without_shortcuts():
    clauses = MODULE.complete_clauses()
    row = MODULE.decode(clauses[0], clauses[-1])
    assert row["rendered"].count(".") == 2
    assert row["audit"]["letters"] > 38
    assert all(value is False for value in row["anti_shortcut_flags"].values())
    assert row["cross_boundary_obligations"]


def test_audit_has_independent_pointer_and_hash_evidence():
    result = MODULE.audit("The baker carries the bell near the garden.")
    assert result["independent_two_pointer_exact"] is False
    assert result["sha256_forward"] != result["sha256_reverse"]
    assert result["two_pointer_mismatches"]


def test_run_records_actual_prose_and_concrete_repair():
    result = MODULE.run()
    assert result["actual_prose"]
    assert result["candidate_count"] == result["stats"]["bounded_pairs"]
    assert result["failure_and_repair"]["concrete_next_repair"]
    out = ROOT / "runs" / (MODULE.EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n")
