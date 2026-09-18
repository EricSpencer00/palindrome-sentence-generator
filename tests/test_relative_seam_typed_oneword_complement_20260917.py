import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.relative_seam_typed_oneword_complement_20260917 import run,letters
def test_oneword_complement_isolated_and_audited():
 p=run(); assert p['stats']['rendered']==64; assert p['stats']['exact']==0
 assert {x['complement_type'] for x in p['rendered_candidates']}=={'deictic','directional'}
 for x in p['rendered_candidates']:
  assert x['bridge_fixed']=='and then' and x['audit']['letters']==len(letters(x['rendered']))
  assert x['audit']['sha256_forward']!=x['audit']['sha256_reverse'] and not x['provenance']['catalogue_used']
