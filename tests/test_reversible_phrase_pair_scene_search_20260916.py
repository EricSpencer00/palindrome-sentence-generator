import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('lane',ROOT/'experiments/reversible_phrase_pair_scene_search_20260916.py');lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)
def test_reversible_phrase_scene():
 lane.main();d=json.loads((ROOT/'runs'/(lane.ID+'.json')).read_text())
 assert all(x['audit']['letters']>38 and x['connected_scene'] for x in d['candidates'])
 assert all(x['audit']['two_pointer_exact'] is False for x in d['candidates'])
 assert d['novelty_preflight']['disconnected_chain'] is False
