import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.paired_determiner_inflection_20260917 import run,letters
def test_determiner_inflection_candidates_are_audited():
 r=run();assert r['stats']['rendered']==4;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['agreement']['left_right_agree'];assert x['provenance']['same_semantic_frame'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
