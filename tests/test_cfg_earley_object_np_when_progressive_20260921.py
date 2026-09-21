import importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).parents[1]; spec=importlib.util.spec_from_file_location("cfg_object_when_prog",ROOT/"experiments/cfg_earley_object_np_when_progressive_20260921.py"); lane=importlib.util.module_from_spec(spec); sys.modules[spec.name]=lane; spec.loader.exec_module(lane)
def test_object_np_when_progressive_enters_before_rendering():
 result=lane.run(); assert result["novelty_preflight"]["passed"]; assert len(result["rows"])==3
 assert result["stats"]["rendered_lengths"]==[139,136,137]
 for row in result["rows"]:
  assert row["letters"]>=39 and row["left_chart"]["accepted"] and row["right_chart"]["accepted"]
  assert row["provenance"]["topology"]=="VP -> V [NP_obj -> Det N CP; when NP Aux V-ing]" and row["exact_check_pointer"]["exact"] is False and row["exact_check_sha"]["exact"] is False
  assert row["independent_exact_agreement"] and row["anti_shortcut_flags"]["post_render_repair"] is False
