import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.seam_paired_relative_boundary_edit_20260918 import run,letters
def test_paired_relative_boundary_edit_audited():
 r=run();assert r['stats']['rendered']==4;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['seam']['paired_relative_edit'] and x['seam']['paired_boundary_edit'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
