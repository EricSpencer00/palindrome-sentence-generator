import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.single_scene_appositive_redeployment_20260918 import run,letters
def test_orthogonal_single_scene_redeployment_audited():
 r=run();assert r['stats']['rendered']==3;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert not x['policy']['two_region_authoring'];assert x['provenance']['novelty_preflight'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
