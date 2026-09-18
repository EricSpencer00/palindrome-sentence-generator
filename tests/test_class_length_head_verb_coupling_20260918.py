import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.class_length_head_verb_coupling_20260918 import run,letters
def test_head_verb_class_gate_and_audit():
 r=run();assert r['stats']['rendered']>0;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['constraint']['class_length_matched'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
