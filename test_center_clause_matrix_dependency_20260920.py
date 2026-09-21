import json
from pathlib import Path
from experiments.center_clause_matrix_dependency_20260920 import audit,main
def test_matrix(monkeypatch,tmp_path):
 monkeypatch.chdir(tmp_path);main();r=json.loads(Path('runs/center-clause-matrix-dependency-20260920.json').read_text())
 assert r['stats']['states']==16 and r['stats']['clause_expansions']==16
 assert r['rejected_traces'] and r['independent_audit']['forward_sha256']
 assert all(x['audit']['two_pointer_checked'] for x in r['complete_prose_controls'])
def test_exact(): assert audit('Able was I ere I saw Elba')['exact']
