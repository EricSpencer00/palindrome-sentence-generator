import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("parse_tree_exact_cover", Path(__file__).parents[1] / "experiments/parse_tree_exact_cover_20260916.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
audit, run, tree = _mod.audit, _mod.run, _mod.tree

def test_tree_has_five_role_spans():
    assert [n.role for n in tree().terminals()] == ["det", "subject", "verb", "object", "adjunct"]

def test_bounded_probes_are_complete_and_long():
    result = run()
    assert result["base"]["probe_count"] == 24
    assert result["base"]["eligible_39_plus"] == 24
    assert result["audits"]["mechanical"]
    assert result["base"]["exact_count"] == 0

def test_independent_mirror_audits_agree_on_known_tape():
    row = audit("a man", "nam a")
    assert row["exact"] and row["two_pointer"] and row["hash_equal"]
