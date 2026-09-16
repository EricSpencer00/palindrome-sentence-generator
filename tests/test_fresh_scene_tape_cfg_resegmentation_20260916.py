import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("lane",ROOT/"experiments/fresh_scene_tape_cfg_resegmentation_20260916.py"); lane=importlib.util.module_from_spec(spec); spec.loader.exec_module(lane)
def test_fresh_scene_resegmentation_and_repair():
 lane.main(); d=json.loads((ROOT/"runs"/(lane.ID+".json")).read_text())
 assert all(x["cfg_parse"]["complete"] for x in d["candidates"])
 assert all(x["audit"]["two_pointer_exact"] is False for x in d["candidates"])
 assert min(x["audit"]["letters"] for x in d["candidates"])>100
 assert d["novelty_preflight"]["fixed_tape_used"] is False and "replaced" in d["provenance"]["repair"]
