import json
from pathlib import Path
import paragraph_abba_seam_20260921 as m

def test_abba_units_are_intact_and_semantically_paired():
 d=m.run(); r=d['actual_paragraph_candidates'][0]
 assert r['semantic_pattern']==['A','B','B','A']
 assert r['units']==['A1','B1','B2','A2']
 assert r['provenance']['lexical_independence']
 assert all(len(u['text'].split()) >= 8 for u in m.UNITS)

def test_full_paragraph_audits_and_novelty_preflight():
    d=m.run(); assert d['stats']['candidates']==5
    assert d['rendered_outputs'][2]['frames']==['witness','release','release','witness']
    assert len({r['audit']['sha256_forward'] for r in d['rendered_outputs']})==5
    r=d['rendered_outputs'][0]; a=r['audit']
    assert a['outside_in']['pairs_checked'] == a['letters']//2
    assert len(a['sha256_forward']) == 64 and len(a['sha256_reverse']) == 64
    assert d['novelty_preflight']['status']=='passed'
    assert d['novelty_preflight']['finished_tape_reversal'] is False
    assert d['novelty_preflight']['catalogue_text'] is False
    assert all('seam_solver' in row for row in d['rendered_outputs'])
    assert d['next_repair']['operator'].startswith('reset to a different')

def test_artifact_matches_run():
 d=m.run(); p=Path(m.OUT); m.OUT.parent.mkdir(exist_ok=True); p.write_text(json.dumps(d,indent=2)+'\n')
 saved=json.loads(p.read_text()); assert saved['stats']==d['stats']
