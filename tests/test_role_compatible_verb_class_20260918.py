import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.role_compatible_verb_class_20260918 import run,letters
def test_role_compatible_verb_class_audited():
 r=run();assert r['stats']['rendered']==4;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['verb_class']['inflection_preserved'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
