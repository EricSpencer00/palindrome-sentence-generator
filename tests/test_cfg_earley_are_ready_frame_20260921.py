import importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).parents[1]; spec=importlib.util.spec_from_file_location("cfg_ready_frame",ROOT/"experiments/cfg_earley_are_ready_frame_20260921.py"); lane=importlib.util.module_from_spec(spec); sys.modules[spec.name]=lane; spec.loader.exec_module(lane)
def test_ready_frame_enters_before_rendering_and_records_lengths():
 result=lane.run(); assert result["novelty_preflight"]["passed"]; assert len(result["rows"])==3
 assert result["stats"]["rendered_lengths"]==[140,144,140]
 for row in result["rows"]:
  assert row["letters"]>=39 and row["left_chart"]["accepted"] and row["right_chart"]["accepted"]
  assert row["provenance"]["frame"]=="are ready to V" and row["exact_check_pointer"]["exact"] is False and row["exact_check_sha"]["exact"] is False
  assert row["independent_exact_agreement"] and row["anti_shortcut_flags"]["post_render_repair"] is False
