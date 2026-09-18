import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.authored_mirror_pair_composition_20260918 import run,letters
def test_authored_mirror_pairs_are_nonself_and_audited():
 r=run();assert r['stats']['rendered']==3;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert not x['provenance']['left_self_palindrome'];assert not x['provenance']['right_self_palindrome'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
