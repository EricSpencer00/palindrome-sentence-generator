import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.appositive_seam_class_constraint_20260918 import run,letters
def test_appositive_class_constraint_audited():
 r=run();assert r['stats']['rendered']>0;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['seam_constraint']['satisfied'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
