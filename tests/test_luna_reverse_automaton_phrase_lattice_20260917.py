import hashlib, importlib.util, json, sys
from pathlib import Path
ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("luna_reverse_automaton_phrase_lattice_20260917", ROOT / "experiments/luna_reverse_automaton_phrase_lattice_20260917.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)

def test_lattice_intersects_at_characters_without_sentence_reverse():
    result = mod.run()
    lattice = result["lattice_intersection"]
    assert lattice["character_boundary_intersection"]
    assert lattice["resegmentation"] and not lattice["finished_sentence_reverse"]
    assert len(result["candidates"][0]["phrase_edges"]) >= 6

def test_exact_witness_is_fail_closed_not_reader_claim():
    row = json.loads(mod.OUT.read_text())["candidates"][0]
    assert row["pointer"]["exact"] and row["sha"]["exact"]
    assert row["letters"] > 38
    assert row["mechanically_admitted"] is False
    assert row["reader_eligible"] is False
    assert any(row["anti_shortcut"].values())

def test_novelty_and_generator_hash_are_recorded():
    result = json.loads(mod.OUT.read_text())
    assert result["novelty_preflight"]["status"] == "passed"
    expected = hashlib.sha256((ROOT / "experiments/luna_reverse_automaton_phrase_lattice_20260917.py").read_bytes()).hexdigest()
    assert result["provenance"]["generator_sha256"] == expected
    assert result["provenance"]["independent_pointer"] and result["provenance"]["independent_sha"]
