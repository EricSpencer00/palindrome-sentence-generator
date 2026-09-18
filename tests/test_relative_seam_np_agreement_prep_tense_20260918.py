import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.relative_seam_np_agreement_prep_tense_20260918 import run,letters
def test_np_agreement_prep_tense_is_bounded_and_audited():
 p=run(); assert p['stats']['rendered']==32; assert p['stats']['exact']==0
 assert {x['frame_type'] for x in p['rendered_candidates']}=={'present_singular_by','past_plural_near'}
 for x in p['rendered_candidates']:
  assert x['audit']['letters']==len(letters(x['rendered'])) and x['audit']['sha256_forward']!=x['audit']['sha256_reverse']
  assert not x['provenance']['catalogue_used']
