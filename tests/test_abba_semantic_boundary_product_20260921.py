import importlib.util
import sys
from pathlib import Path

P = Path(__file__).parents[1] / "experiments/abba_semantic_boundary_product_20260921.py"
spec = importlib.util.spec_from_file_location("abba_product", P)
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)

def test_product_has_independent_prose_controls_and_live_pruning():
    d = mod.run()
    assert d["stats"]["complete_products"] == 81
    assert d["stats"]["pruned_at_boundary"] > 0
    assert all(len(mod.letters(x["rendered"])) > 38 for x in d["controls"])
    assert all(x["provenance"]["finished_tape_reversal"] is False for x in d["controls"])

def test_audit_is_independent_and_no_fabricated_exact_candidate():
    d = mod.run()
    for row in d["exact_candidates"]:
        assert row["audit"]["two_pointer_exact"]
        assert row["audit"]["sha256_forward"] == row["audit"]["sha256_reverse_obligation"]
    assert d["stats"]["exact_over_38"] == 0
