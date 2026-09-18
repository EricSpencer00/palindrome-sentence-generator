import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.relative_seam_threeway_locative_length_20260918 import run,letters
def test_threeway_locative_length_is_bounded_and_audited():
 p=run(); assert p['stats']['rendered']==16; assert p['stats']['exact']==0
 assert {x['frame_type'] for x in p['rendered_candidates']}=={'short_singular','expanded_plural'}
 for x in p['rendered_candidates']:
  assert x['audit']['letters']==len(letters(x['rendered'])) and x['audit']['sha256_forward']!=x['audit']['sha256_reverse']
  assert not x['provenance']['catalogue_used']
