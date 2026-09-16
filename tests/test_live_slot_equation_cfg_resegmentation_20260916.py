import importlib.util, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("live_slot",ROOT/"experiments/live_slot_equation_cfg_resegmentation_20260916.py")
mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)

def test_novelty_preflight_and_run():
    result=mod.run()
    assert result["novelty_preflight"]["status"]=="passed"
    assert result["source"]["provenance"].startswith("two independently authored")
    assert result["rendered_candidates"]
    assert all("independent_sha256" in x for x in result["rendered_candidates"])
    assert result["next_repair"]["status"]=="required"
    repair=result["bounded_repair"]
    assert repair["operator"].startswith("single-heldout")
    assert repair["audit"]["independent_sha256"]
    assert repair["audit"]["mechanically_admitted"] is False
    assert repair["provenance"].startswith("held-out")
    second=result["second_bounded_repair"]
    assert second["operator"].startswith("single-heldout-object")
    assert second["audit"]["independent_sha256"]
    assert second["audit"]["mechanically_admitted"] is False

def test_slot_equation_is_not_word_order_symmetry():
    source=mod.construct_slot_equation()
    assert source["left_rendered"] != source["right_rendered"]
    assert source["obligations"]
    assert source["provenance"].endswith("no corpus import")
