import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.grammar_dp_mirror_pair_center_20260918 import run,letters
def test_grammar_dp_route_is_bounded_and_audited():
 p=run(); assert p['stats']['rendered']==8; assert p['stats']['exact']==0
 for x in p['rendered_candidates']:
  assert x['dp_state']['left_authored_independently'] and x['dp_state']['right_authored_independently']
  assert x['audit']['letters']==len(letters(x['rendered'])) and x['audit']['sha256_forward']!=x['audit']['sha256_reverse']
  assert not x['provenance']['catalogue_used'] and not x['provenance']['finished_tape_reversal']
