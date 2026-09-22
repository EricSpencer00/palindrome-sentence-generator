import json
from pathlib import Path
import experiments.paragraph_aabc_reset_20260921 as m

def test_fresh_aabc():
 d=m.run(); r=d['actual_paragraph_candidates'][0]
 assert r['semantic_pattern']==['A','A','B','C']; assert r['roles']==['departure','return','setting','resolution']
 assert r['provenance']['prior_topology_reuse'] is False and len(set(u['text'] for u in m.UNITS))==4

def test_audits_and_novelty():
 d=m.run(); a=d['rendered_outputs'][0]['audit']; assert a['pairs_checked']==a['letters']//2
 assert len(a['sha256_forward'])==64 and a['two_pointer_exact']==a['sha_equal']; assert d['novelty_preflight']['status']=='passed'
 assert all(d['novelty_preflight'][k] is False for k in ('abba_reuse','abac_reuse','abca_reuse','abcb_reuse'))

def test_roundtrip():
 d=m.run(); Path(m.OUT).write_text(json.dumps(d,indent=2)+'\n'); assert json.loads(Path(m.OUT).read_text())['stats']==d['stats']
