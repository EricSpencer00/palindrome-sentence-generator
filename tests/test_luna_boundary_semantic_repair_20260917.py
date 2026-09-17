import importlib.util, json
from pathlib import Path
ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("luna_boundary", ROOT / "experiments/luna_boundary_semantic_repair_20260917.py")
lane = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = lane
spec.loader.exec_module(lane)

def test_typed_lattice_and_pre_render_obligation_produce_complete_scene():
    lane.main(); report = json.loads((ROOT / "runs" / (lane.ID + ".json")).read_text())
    assert report["stats"]["typed_states"] == 32
    assert report["stats"]["rendered"] == 3
    assert all(row["boundary_obligation"]["obligation"] for row in report["candidates"])
    assert all(row["audit"]["letters"] > 38 for row in report["candidates"])
    assert all(row["audit"]["two_pointer"]["equal"] is False for row in report["candidates"])
    assert report["novelty_preflight"]["fixed_tape_used"] is False
    assert report["novelty_preflight"]["word_order_mirror_used"] is False

def test_valency_rejects_non_transitive_verb_and_pointer_is_independent():
    bad = lane.Lexeme("waits", "verb", "intransitive")
    assert lane.typed_scene(lane.AGENTS[0], bad, lane.OBJECTS[0], lane.PLACES[0], lane.CONSEQUENCES[0]) is None
    pointer = lane.independent_pointer("The night porter checks the brass key beside the locked boathouse, and finds the waiting boat.")
    assert pointer["letters"] > 38 and pointer["equal"] is False
