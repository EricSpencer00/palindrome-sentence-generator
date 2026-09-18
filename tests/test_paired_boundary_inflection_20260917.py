import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.paired_boundary_inflection_20260917 import run,letters
def test_boundary_inflections_preserve_attachment_and_audit():
 r=run();assert r['stats']['rendered']==4;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['boundary_inflection']['attachment']=='location';assert x['provenance']['semantic_role_keys_preserved'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
