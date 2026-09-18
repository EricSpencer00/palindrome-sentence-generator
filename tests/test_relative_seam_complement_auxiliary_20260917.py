import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.relative_seam_complement_auxiliary_20260917 import run,letters
def test_complement_auxiliary_is_bounded_and_audited():
 p=run(); assert p['stats']['rendered']==64; assert p['stats']['exact']==0
 assert {x['pair_type'] for x in p['rendered_candidates']}=={'deictic_modal','directional_future'}
 for x in p['rendered_candidates']:
  assert x['bridge_fixed']=='and then' and x['audit']['letters']==len(letters(x['rendered']))
  assert x['audit']['sha256_forward']!=x['audit']['sha256_reverse'] and not x['provenance']['catalogue_used']
