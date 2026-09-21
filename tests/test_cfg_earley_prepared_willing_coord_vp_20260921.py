import importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).parents[1]; spec=importlib.util.spec_from_file_location("cfg_coord_vp",ROOT/"experiments/cfg_earley_prepared_willing_coord_vp_20260921.py"); lane=importlib.util.module_from_spec(spec); sys.modules[spec.name]=lane; spec.loader.exec_module(lane)
def test_coordinated_vp_enters_before_rendering_and_records_lengths():
 result=lane.run(); assert result["novelty_preflight"]["passed"]; assert len(result["rows"])==3
 assert result["stats"]["rendered_lengths"]==[166,170,166]
 for row in result["rows"]:
  assert row["letters"]>=39 and row["left_chart"]["accepted"] and row["right_chart"]["accepted"]
  assert row["provenance"]["vp_topology"]=="prepared and willing coordinated VP" and row["exact_check_pointer"]["exact"] is False and row["exact_check_sha"]["exact"] is False
  assert row["independent_exact_agreement"] and row["anti_shortcut_flags"]["post_render_repair"] is False
