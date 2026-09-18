import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.paired_opposing_noun_seam_20260917 import run,letters
def test_paired_opposing_nouns_are_audited():
 r=run();assert r['stats']['rendered']==4;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['seam_selection']['coordinated_same_seam'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
