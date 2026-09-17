import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('lane',ROOT/'experiments/feature_carrying_center_cfg_20260916.py');lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)
def test_feature_cfg_free_center():
 lane.main();d=json.loads((ROOT/'runs'/(lane.ID+'.json')).read_text())
 assert all(x['audit']['letters']>38 for x in d['candidates'])
 assert all(x['features']['center']=='free' and x['audit']['two_pointer_exact'] is False for x in d['candidates'])
 assert d['novelty_preflight']['posthoc_reversal'] is False
