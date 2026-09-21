import json
from pathlib import Path
from experiments.center_dependency_relation_graph_20260920 import audit,main
def test_dependency_graph(monkeypatch,tmp_path):
 monkeypatch.chdir(tmp_path);main();r=json.loads(Path('runs/center-dependency-relation-graph-20260920.json').read_text())
 assert r['stats']['states']==256 and r['stats']['dependency_expansions']==256
 assert r['rejected_traces'] and r['independent_audit']['forward_sha256']
 assert all(x['audit']['two_pointer_checked'] for x in r['complete_prose_controls'])
def test_exact(): assert audit('Able was I ere I saw Elba')['exact']
