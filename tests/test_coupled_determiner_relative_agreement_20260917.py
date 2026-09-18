import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.coupled_determiner_relative_agreement_20260917 import run,letters
def test_coupled_agreement_is_audited():
 r=run();assert r['stats']['rendered']==4;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['agreement']['subject_and_relative_agree'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
