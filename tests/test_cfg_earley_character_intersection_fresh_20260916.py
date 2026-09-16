import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("lane",ROOT/"experiments/cfg_earley_character_intersection_fresh_20260916.py")
lane=importlib.util.module_from_spec(spec); spec.loader.exec_module(lane)
def test_fresh_cfg_prose_audit():
 lane.main(); d=json.loads((ROOT/"runs"/(lane.ID+".json")).read_text())
 assert [x["audit"]["letters"] for x in d["candidates"]]==[83,135]
 assert all(x["audit"]["two_pointer_exact"] is False for x in d["candidates"])
 assert d["novelty_preflight"]["fixed_tape_used"] is False
