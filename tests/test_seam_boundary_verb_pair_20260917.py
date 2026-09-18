import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.seam_boundary_verb_pair_20260917 import run,letters
def test_seam_boundary_verb_pair_audited():
 r=run();assert r['stats']['rendered']==4;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['live_seam']['paired_boundary'];assert x['provenance']['role_compatible_verb_edits'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
