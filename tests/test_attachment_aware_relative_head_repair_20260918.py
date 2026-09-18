import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.attachment_aware_relative_head_repair_20260918 import run,letters
def test_attachment_aware_relative_heads_are_audited():
 r=run();assert r['stats']['rendered']==4;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['attachment']['same_role'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
