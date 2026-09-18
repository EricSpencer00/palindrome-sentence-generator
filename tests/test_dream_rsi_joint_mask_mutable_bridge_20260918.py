import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.dream_rsi_joint_mask_mutable_bridge_20260918 import run,letters
def test_joint_mask_bridge_route_is_fresh_and_audited():
 p=run(); assert p['stats']['rendered']==4; assert p['stats']['exact']==0
 for x in p['rendered_candidates']:
  assert x['mask_policy']['mutable_bridge'] and len(x['regions'])==4
  assert x['audit']['letters']==len(letters(x['rendered'])) and x['audit']['sha256_forward']!=x['audit']['sha256_reverse']
  assert not x['provenance']['catalogue_used']
