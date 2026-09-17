import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('lane',ROOT/'experiments/boundary_shift_scene_equation_lattice_20260916.py');lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)
def test_boundary_shift_lattice():
 lane.main();d=json.loads((ROOT/'runs'/(lane.ID+'.json')).read_text())
 assert len(d['candidates'])==2 and all(x['audit']['letters']>100 for x in d['candidates'])
 assert all(x['boundary_shift']['different_counts'] and x['audit']['two_pointer_exact'] is False for x in d['candidates'])
 assert d['novelty_preflight']['word_order_mirror'] is False
