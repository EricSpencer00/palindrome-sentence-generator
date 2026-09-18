import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.live_seam_paired_lexical_repair_20260918 import run,letters
def test_live_seam_paired_repair_is_audited():
 r=run();assert r['stats']['rendered']==4;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['live_seam']['base_first_mismatch'] is not None;assert x['provenance']['paired_lexical_repair'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
