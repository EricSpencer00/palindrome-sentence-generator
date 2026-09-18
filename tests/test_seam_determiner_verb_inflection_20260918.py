import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.seam_determiner_verb_inflection_20260918 import run,letters
def test_determiner_verb_inflection_audited():
 r=run();assert r['stats']['rendered']==4;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['agreement']['determiner_verb_coupled'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
