import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.live_seam_boundary_growth_20260917 import run,letters
def test_live_boundary_growth_records_seam_and_audit():
 r=run();assert r['stats']['rendered']==4;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['live_seam']['paired_boundary'];assert x['provenance']['live_seam_audit'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
