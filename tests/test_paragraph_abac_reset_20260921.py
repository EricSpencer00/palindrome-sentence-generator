import json
from pathlib import Path
import experiments.paragraph_abac_reset_20260921 as m

def test_fresh_abac_topology():
 d=m.run(); assert d['stats']['candidates']==4; r=d['actual_paragraph_candidates'][0]
 assert r['semantic_pattern']==['A','B','A','C']
 assert r['roles']==['observation','transit','observation','archival']
 assert r['provenance']['abba_output_reuse'] is False
 assert len(set(u['text'] for u in m.UNITS))==4
 assert d['actual_paragraph_candidates'][1]['roles']==['calibration','negotiation','calibration','cultivation']

def test_live_exact_and_novelty_audits():
 d=m.run(); assert len({r['audit']['sha256_forward'] for r in d['rendered_outputs']})==4; a=d['rendered_outputs'][0]['audit']
 assert a['pairs_checked']==a['letters']//2
 assert len(a['sha256_forward'])==64 and len(a['sha256_reverse'])==64
 assert d['novelty_preflight']['status']=='passed'
 assert d['novelty_preflight']['reused_abba_units'] is False
 assert a['two_pointer_exact'] == (a['sha256_forward']==a['sha256_reverse'])
 assert d['repair_attempts'][0]['provenance']['repair_pass'].startswith('joint repeated-A')

def test_artifact_round_trip():
 d=m.run(); Path(m.OUT).write_text(json.dumps(d,indent=2)+'\n')
 assert json.loads(Path(m.OUT).read_text())['stats']==d['stats']
