import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.dream_rsi_three_region_policy_scene_20260918 import run,letters
def test_policy_scene_is_fresh_bounded_and_audited():
 p=run(); assert p['stats']['rendered']==2; assert p['stats']['exact']==0
 assert {x['policy_branch'] for x in p['rendered_candidates']}=={'maritime','civic'}
 for x in p['rendered_candidates']:
  assert len(x['regions'])==3 and x['audit']['letters']==len(letters(x['rendered']))
  assert x['audit']['sha256_forward']!=x['audit']['sha256_reverse']
  assert not x['provenance']['catalogue_used']
