import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "experiments/edge_lexical_compatibility_20260920.py"
spec = importlib.util.spec_from_file_location("edge_lane", P)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_edge_lane_is_bounded_and_joint():
    x = m.run()
    assert x["inventory"]["opening_phrases"] >= 32
    assert x["inventory"]["closing_phrases"] >= 32
    assert x["inventory"]["pairings"] == x["inventory"]["opening_phrases"] * x["inventory"]["closing_phrases"]
    assert x["novelty_preflight"]["finished_tape_reversal"] is False
    assert x["novelty_preflight"]["post_hoc_repair"] is False

def test_seed_edge_keeps_full_residual_and_audits_controls():
    x = m.run()
    hit = [r for r in x["deepest_valid_residuals"] if r["opening"] == "An aide rips nine memos" and r["closing"] == "men inspire Diana"]
    assert hit and hit[0]["left"] == "emos" and hit[0]["right"] == ""
    assert all(c["audit"]["sha_equal"] == c["audit"]["exact"] for c in x["controls"])
    assert not x["exact_candidates"]
