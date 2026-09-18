import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.agreement_relative_seam_length_20260918 import run,letters
def test_agreement_relative_seam_length_audited():
 r=run();assert r['stats']['rendered']>0;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['balance']['equal'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
