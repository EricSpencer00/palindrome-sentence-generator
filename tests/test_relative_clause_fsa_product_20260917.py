import importlib.util, json, sys
from pathlib import Path
ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("relative_clause_fsa_product", ROOT / "experiments/relative_clause_fsa_product_20260917.py")
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)

def test_withheld_control_and_exact_long_search():
    result = mod.run()
    assert result["status"] == "completed_no_admitted_closure"
    assert result["withheld_control"]["independent_audit"]["exact"]
    assert result["candidates"] == []
    assert result["mechanically_admitted"] is False

def test_artifact_provenance_and_fresh_route():
    result = json.loads(mod.OUT.read_text())
    assert result["method"].endswith("relative_clause")
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["provenance"]["seed_used"] is False
    assert all("flowwar" not in word for word in result["search"]["lexical_inventory"])
