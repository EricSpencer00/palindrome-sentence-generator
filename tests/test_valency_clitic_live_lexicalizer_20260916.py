import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('lane',ROOT/'experiments/valency_clitic_live_lexicalizer_20260916.py');lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)
def test_valency_clitic_candidates():
 lane.main();d=json.loads((ROOT/'runs'/(lane.ID+'.json')).read_text())
 assert all(x['audit']['letters']>38 and x['semantic_consistency'] for x in d['candidates'])
 assert all(x['audit']['two_pointer_exact'] is False for x in d['candidates'])
 assert d['novelty_preflight']['fixed_tape_used'] is False
