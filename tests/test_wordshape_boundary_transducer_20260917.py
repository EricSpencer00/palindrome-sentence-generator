import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("wordshape", Path(__file__).parents[1] / "experiments/wordshape_boundary_transducer_20260917.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
generate = mod.generate

def test_wordshape_is_live_product_and_exhausts():
    r=generate(); assert r["status"] == "exhausted"; assert r["config"]["simultaneous_lexical_generation"]
    assert not r["config"]["fixed_tape"] and not r["config"]["posthoc_reverse"]

def test_exact_closure_and_preflight_are_explicit():
    r=generate(); assert all(x["independent_exact_audit"]["exact"] for x in r["closures"])
    assert r["novelty_preflight"]["excluded"]["brown_reverse_segmentation"]
