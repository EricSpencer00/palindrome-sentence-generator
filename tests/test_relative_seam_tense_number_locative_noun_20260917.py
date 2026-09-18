import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.relative_seam_tense_number_locative_noun_20260917 import run,letters
def test_tense_number_locative_noun_is_bounded_and_audited():
 p=run(); assert p['stats']['rendered']==32; assert p['stats']['exact']==0
 assert {x['frame_type'] for x in p['rendered_candidates']}=={'present_singular','past_plural'}
 for x in p['rendered_candidates']:
  assert x['audit']['letters']==len(letters(x['rendered'])) and x['audit']['sha256_forward']!=x['audit']['sha256_reverse']
  assert not x['provenance']['catalogue_used']
