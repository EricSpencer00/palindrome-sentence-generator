import json
from pathlib import Path
import paragraph_aabc_transducer_20260921 as m

def test_a_first_transducer_and_controls():
 d=m.run(); assert d['stats']['candidates']==12; r=d['actual_paragraph_candidates'][0]
 assert r['live_transducer']['phase_order']==['choose_A1','choose_C_endpoint','choose_A2','fill_B']
 assert r['provenance']['joint_A_and_C_endpoint_choice_before_B']
 assert all(x['live_transducer']['endpoint_compatibility_class']=='endpoint-a' for x in d['rendered_outputs'])
 assert all(len(x['rendered'].split('.'))>=4 for x in d['rendered_outputs'])

def test_exact_pointer_sha_novelty():
 d=m.run(); assert d['novelty_preflight']['status']=='passed'
 for r in d['rendered_outputs']:
  a=r['audit']; assert a['pairs_checked']==a['letters']//2; assert a['two_pointer_exact']==a['sha_equal']
 assert d['stats']['exact']==0 and d['residual_seam_key'] is not None

def test_artifact():
 d=m.run(); Path(m.OUT).write_text(json.dumps(d,indent=2)+'\n'); assert json.loads(Path(m.OUT).read_text())['stats']==d['stats']
