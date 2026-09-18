import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.relative_seam_locative_determiner_20260917 import run,letters
def test_locative_determiner_is_bounded_and_audited():
 p=run(); assert p['stats']['rendered']==64; assert p['stats']['exact']==0
 assert {x['determiner_type'] for x in p['rendered_candidates']}=={'short_determiner','expanded_determiner'}
 for x in p['rendered_candidates']:
  assert x['audit']['letters']==len(letters(x['rendered'])) and x['audit']['sha256_forward']!=x['audit']['sha256_reverse']
  assert not x['provenance']['catalogue_used']
