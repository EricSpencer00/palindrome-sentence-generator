import json
from pathlib import Path
import paragraph_abcb_reset_20260921 as m

def test_fresh_abcb():
 d=m.run(); assert d['stats']['candidates']==3; r=d['actual_paragraph_candidates'][0]
 assert r['semantic_pattern']==['A','B','C','B']; assert r['roles']==['departure','exchange','setting','exchange']
 assert r['provenance']['prior_topology_reuse'] is False and len(set(u['text'] for u in m.UNITS))==4

def test_audits_and_novelty():
 d=m.run(); assert len(d['repair_attempts'])==2; assert len({r['audit']['sha256_forward'] for r in d['rendered_outputs']})==3; a=d['rendered_outputs'][0]['audit']; assert a['pairs_checked']==a['letters']//2
 assert len(a['sha256_forward'])==64 and a['two_pointer_exact']==a['sha_equal']; assert d['novelty_preflight']['status']=='passed'
 assert d['novelty_preflight']['abba_reuse'] is False

def test_roundtrip():
 d=m.run(); Path(m.OUT).write_text(json.dumps(d,indent=2)+'\n'); assert json.loads(Path(m.OUT).read_text())['stats']==d['stats']
