import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.inflectional_same_seam_20260917 import run,letters
def test_inflectional_same_class_candidates_are_audited():
 r=run();assert r['stats']['rendered']>0;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['provenance']['same_seam_class'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
