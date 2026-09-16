import importlib.util, json
from pathlib import Path
ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("lane", ROOT / "experiments/lexical_word_equation_grammar_intersection_20260916.py")
lane = importlib.util.module_from_spec(spec); spec.loader.exec_module(lane)

def test_heldout_prose_has_independent_audit_and_equation():
    lane.main()
    data = json.loads((ROOT / "runs" / f"{lane.ID}.json").read_text())
    assert len(data["candidates"]) == 4
    assert all(c["semantic_consistency"] and c["audit"]["two_pointer_exact"] is False for c in data["candidates"])
    assert all("residual" in c["equation"] for c in data["candidates"])
    assert data["novelty_preflight"]["fixed_tape_used"] is False
