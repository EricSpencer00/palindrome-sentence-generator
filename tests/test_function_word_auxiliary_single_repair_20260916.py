import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('lane',ROOT/'experiments/function_word_auxiliary_single_repair_20260916.py');lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)
def test_single_auxiliary_repair():
 lane.main();d=json.loads((ROOT/'runs'/(lane.ID+'.json')).read_text());x=d['candidates'][0]
 assert d['stats']['rendered']==1 and x['audit']['letters']>38 and x['audit']['two_pointer_exact'] is False
 assert x['past_participle_fixed']=='stored' and x['agreement_preserved'] and d['novelty_preflight']['sweep'] is False
