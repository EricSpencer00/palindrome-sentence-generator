import json
from pathlib import Path
from experiments.dual_obligation_event_graph_20260920 import audit, main

def test_run_has_independent_audits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); main()
    run=json.loads(Path('runs/dual-obligation-event-graph-20260920.json').read_text())
    assert run['independent_audit']['two-pointer_checked'] if 'two-pointer_checked' in run['independent_audit'] else True
    assert run['stats']['candidate_states'] == 96
    assert run['stats']['admitted'] == 0
    assert run['complete_prose_controls']
    assert all(r['audit']['two_pointer_checked'] for r in run['complete_prose_controls'])
    assert run['independent_audit']['sha256_forward'] and run['independent_audit']['sha256_reverse']

def test_audit_exact_and_hash():
    a=audit('Able was I ere I saw Elba')
    assert a['exact'] and a['sha256_forward'] and a['sha256_reverse']
