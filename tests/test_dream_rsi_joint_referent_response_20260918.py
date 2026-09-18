import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.dream_rsi_joint_referent_response_20260918 import run,letters
def test_joint_mask_route_is_fresh_and_audited():
 p=run(); assert p['stats']['rendered']==2; assert p['stats']['exact']==0
 for x in p['rendered_candidates']:
  assert len(x['regions'])==3 and x['mask_policy']['joint_referent_response']
  assert x['audit']['letters']==len(letters(x['rendered']))
  assert x['audit']['sha256_forward']!=x['audit']['sha256_reverse']
  assert not x['provenance']['catalogue_used']
