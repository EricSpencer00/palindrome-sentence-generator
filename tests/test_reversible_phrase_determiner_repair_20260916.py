import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('lane',ROOT/'experiments/reversible_phrase_determiner_repair_20260916.py');lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)
def test_single_determiner_child():
 lane.main();d=json.loads((ROOT/'runs'/(lane.ID+'.json')).read_text());x=d['candidates'][0]
 assert d['stats']['rendered']==1 and x['audit']['letters']>38 and x['audit']['two_pointer_exact'] is False
 assert x['preposition_fixed']=='beside' and x['location_fixed']=='old quay' and x['valency_preserved']
