import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("lane",ROOT/"experiments/word_boundary_semantic_inflection_dp_20260916.py"); lane=importlib.util.module_from_spec(spec); spec.loader.exec_module(lane)
def test_joint_dp_emits_prose_and_audits():
 lane.main(); d=json.loads((ROOT/"runs"/(lane.ID+".json")).read_text())
 assert all(x["audit"]["letters"]>100 for x in d["candidates"])
 assert all(x["audit"]["two_pointer_exact"] is False for x in d["candidates"])
 assert d["novelty_preflight"]["fixed_tape_used"] is False
