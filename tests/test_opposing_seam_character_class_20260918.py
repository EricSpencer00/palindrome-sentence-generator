import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.opposing_seam_character_class_20260918 import run,letters
def test_opposing_character_class_filter_and_audit():
 r=run();assert r['stats']['rendered']>0;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['class_constraint']['same_vowel_class'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
