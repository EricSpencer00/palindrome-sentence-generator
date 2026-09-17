import importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("agreement",ROOT/"experiments/luna_agreement_morphology_boundary_20260917.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def test_run_has_complete_long_candidates_and_independent_audits():
    out=mod.run()
    assert out["novelty_preflight"]["status"]=="passed"
    assert out["stats"]["over_100"] == out["candidate_count"]
    assert out["candidate_count"] == 3*4*4*3
    row=out["rendered_candidates"][0]
    assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
    assert row["anti_shortcut_flags"] == {"finished_tape_reversal":False,"word_order_symmetry":False,"catalogue_text":False,"fragment":False}
