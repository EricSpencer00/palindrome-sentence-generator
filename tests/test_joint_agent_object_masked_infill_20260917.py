import importlib.util
from pathlib import Path
p=Path(__file__).parents[1];s=importlib.util.spec_from_file_location('x',p/'experiments/joint_agent_object_masked_infill_20260917.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_joint_infill_is_bounded_and_changes_boundaries():
 x=m.run();assert x['stats']['rendered']==16 and x['stats']['longest_letters']>38
 assert all(r['provenance']['span_boundaries_altered'] for r in x['rendered_candidates'])
 assert len({r['rendered'] for r in x['rendered_candidates']})==16
 for r in x['rendered_candidates']:assert r['agreement']['cross_side_number_equal'] and r['audit']['sha256_forward']!=r['audit']['sha256_reverse']
def test_route_exhaustion_is_recorded():
 x=m.run();assert x['stats']['exact']==0 and x['next_repair']['route_exhausted']
