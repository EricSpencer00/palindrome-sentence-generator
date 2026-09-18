import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.appositive_head_verb_seam_20260918 import run,letters
def test_new_appositive_head_verb_seam_audited():
 r=run();assert r['stats']['rendered']==3;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['seam_operator']['new_seed'];assert x['provenance']['novelty_preflight'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
