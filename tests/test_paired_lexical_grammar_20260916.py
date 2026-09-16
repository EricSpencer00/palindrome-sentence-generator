import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("plg", Path(__file__).parents[1]/"experiments/paired_lexical_grammar_20260916.py")
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
def test_joint_pairs_have_dual_audits_and_prose():
    m.main(); p = __import__('json').loads((Path(__file__).parents[1]/"runs/paired-lexical-grammar-20260916.json").read_text())
    assert p["pairs_examined"] == 6 and p["exact_count"] == 0
    assert all(r["audit"]["two_pointer"] == r["audit"]["exact"] for r in p["candidates"])
    assert all(r["provenance"]["fresh_semantic_frames"] for r in p["candidates"])
