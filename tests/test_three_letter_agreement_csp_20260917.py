import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.three_letter_agreement_csp_20260917 import run,letters
def test_three_letter_agreement_filter_and_audit():
 r=run();assert r['stats']['rejected_pre_render']>0;assert r['stats']['rendered']>0;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['csp']['signature_and_number_satisfied'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
