import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.paired_noun_relative_verb_seam_20260917 import run,letters
def test_paired_noun_verb_seam_is_audited():
 r=run();assert r['stats']['rendered']==4;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['coupling']['same_seam'];assert x['provenance']['agreement_preserved'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
