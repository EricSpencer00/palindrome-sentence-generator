import json
from pathlib import Path
import importlib.util

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("experiment", ROOT / "experiments/hand_authored_function_edit_extension_20260916.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def test_every_rendered_state_has_independent_audits_and_is_novel():
    mod.main()
    data = json.loads((ROOT / "runs" / f"{mod.ID}.json").read_text())
    assert data["novelty_preflight"]["exact_signature_collision"] is False
    assert len(data["candidates"]) == 5
    assert all(c["semantic_consistency"] and c["audit"]["two_pointer_exact"] is False for c in data["candidates"])
    assert all(c["audit"]["anti_shortcut"]["seed_wrapped_or_repeated"] is False for c in data["candidates"])
