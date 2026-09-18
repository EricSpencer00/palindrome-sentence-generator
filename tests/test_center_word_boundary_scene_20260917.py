import json
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("center_scene", "experiments/center_word_boundary_scene_20260917.py")
mod = module_from_spec(spec); spec.loader.exec_module(mod)
search = mod.search

def test_online_scene_audit_and_center_word():
 r=search(); assert r['candidate_count']>0; assert r['exact_count']==0
 assert r['stats']['online_prunes']>0
 assert r['stats']['midpoint_inside_token']>0
 for row in r['rendered_candidates']:
  assert row['audit']['independent_two_pointer_exact']==row['audit']['exact']
  assert row['anti_shortcut_flags']['finished_tape_reversal'] is False
  assert row['provenance']['filter_before_final_render'] is True
