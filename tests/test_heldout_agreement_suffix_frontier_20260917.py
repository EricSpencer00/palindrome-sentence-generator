import importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("repair",ROOT/"experiments/heldout_agreement_suffix_frontier_20260917.py")
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
def test_heldout_repair_emits_long_prose_and_suffix_crossing():
 out=mod.run(); assert out["candidate_count"]==3*2*2*3*2; assert out["exact_count"]==0; assert out["stats"]["longest_letters"]>=100; assert out["stats"]["suffix_crossing_rows"]>0
 row=out["rendered_candidates"][0]; assert row["audit"]["sha256_forward"]!=row["audit"]["sha256_reverse"]; assert all(v is False for v in row["anti_shortcut_flags"].values())
def test_feature_conditioned_surface_forms():
 row=mod.realize(mod.FRAMES[0],"singular","present",mod.TAILS[0],"while"); assert row["choices"]["surface_verb"]=="examines"; assert row["morphology_trace"][0]["state"]=="AGREEMENT_FEATURE"
