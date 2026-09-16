import importlib.util
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("mcf", Path(__file__).parents[1] / "experiments/min_cost_flow_clause_realizer_20260916.py")
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
def test_residual_flow_exact_and_debt():
    ok = m.flow_audit("ab", "ba")
    assert ok["exact"] and ok["residual_debt"] == 0 and ok["independent_audit"]
    miss = m.flow_audit("ab", "ca")
    assert not miss["exact"] and miss["residual_debt"] > 0
def test_typed_run_has_intact_prose_and_heldout_repair():
    payload = m.run()
    assert payload["stats"]["min_letters_seen"] >= 39
    assert payload["candidates"][0]["rendered"].endswith(".")
    assert payload["candidates"][1]["label"] == "held-out-repair"
    assert payload["provenance"]["catalogue_imported"] is False
