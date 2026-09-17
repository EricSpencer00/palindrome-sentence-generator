import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('lane',ROOT/'experiments/corpus_seam_reauthored_repair_20260916.py');lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)
def test_single_reauthored_seam_state():
 lane.main();d=json.loads((ROOT/'runs'/(lane.ID+'.json')).read_text());x=d['candidates'][0]
 assert d['stats']['rendered']==1 and x['fresh_authoring'] and x['audit']['letters']>100
 assert x['audit']['two_pointer_exact'] is False and d['novelty_preflight']['replayed_prior_scene'] is False
