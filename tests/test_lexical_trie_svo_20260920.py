from experiments.lexical_trie_svo_20260920 import Trie, audit, consume, paths, run

def test_trie_conditions_edges():
    t=Trie(); t.add("an"); t.add("a"); t.add("the")
    assert "an" in t.compatible("an")
    assert "the" not in t.compatible("an")

def test_live_residual():
    assert consume("abc","ab")==("c","")
    assert consume("abc","ax") is None

def test_variable_svo_paths_and_audits():
    assert len({len(p) for p in paths()})>1
    d=run(12000)
    for row in d["exact_candidates"]: assert row["audit"]==audit(row["rendered"])
