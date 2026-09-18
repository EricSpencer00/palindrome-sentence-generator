import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.coupled_inflectional_head_seam_20260917 import run,letters
def test_coupled_head_key_is_pre_render_and_audited():
 r=run();assert r['stats']['rejected_pre_render']>0;assert r['stats']['rendered']>0;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['coupling']['satisfied'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
