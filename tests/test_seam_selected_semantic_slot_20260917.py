import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from experiments.seam_selected_semantic_slot_20260917 import run,letters
def test_seam_selected_slots_are_independently_audited():
 r=run();assert r['stats']['rendered']==4;assert r['stats']['exact']==0
 for x in r['rendered_candidates']:
  assert x['seam_selection']['base_first_mismatch_index'] is not None;assert x['provenance']['agreement_preserved'];t=letters(x['rendered']);assert x['audit']['two_pointer_exact']==(t==t[::-1])
