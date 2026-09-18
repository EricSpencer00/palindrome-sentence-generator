import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.tiny_joint_noun_verb_boundary_lattice_20260917 import run,letters
def test_joint_lattice_live_rejection_and_audit():
 r=run();assert r['stats']['considered']==4;assert r['stats']['rendered']>0;assert r['stats']['rejected_live']>0;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['live_rejection']['admitted'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
