import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.relative_seam_two_word_bridge_20260917 import run,letters
def test_two_word_bridge_is_bounded_and_audited():
 p=run(); assert p['stats']['rendered']==64; assert p['stats']['exact']==0
 assert {x['bridge_type'] for x in p['rendered_candidates']}=={'coordinating','causal_temporal'}
 for x in p['rendered_candidates']:
  assert len(x['regions'])==4 and x['audit']['letters']==len(letters(x['rendered']))
  assert x['audit']['sha256_forward']!=x['audit']['sha256_reverse']
  assert not x['provenance']['catalogue_used']
