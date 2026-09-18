import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.relative_seam_agreement_20260917 import run,letters
def test_relative_seam_lane_is_bounded_and_audited():
 p=run(); assert p['stats']['rendered']==32; assert p['stats']['exact']==0
 assert {x['attachment'] for x in p['rendered_candidates']}=={'subject_relative','object_relative'}
 for x in p['rendered_candidates']:
  assert x['audit']['letters']==len(letters(x['rendered']))
  assert x['audit']['sha256_forward']!=x['audit']['sha256_reverse']
  assert not x['provenance']['catalogue_used']
