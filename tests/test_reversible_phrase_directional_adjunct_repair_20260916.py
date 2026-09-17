import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('lane',ROOT/'experiments/reversible_phrase_directional_adjunct_repair_20260916.py');lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)
def test_directional_repair_state():
 lane.main();d=json.loads((ROOT/'runs'/(lane.ID+'.json')).read_text())
 assert all(x['audit']['letters']>38 and x['valency_preserved'] for x in d['candidates'])
 assert all(x['audit']['two_pointer_exact'] is False for x in d['candidates'])
 assert all(x['parent_state']=='reversible-phrase-pair-role-repair-20260916' for x in d['candidates'])
