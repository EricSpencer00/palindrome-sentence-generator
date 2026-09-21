import json
from pathlib import Path
from experiments.reverse_morphology_residual_20260920 import audit,main
def test_reverse_lane(monkeypatch,tmp_path):
 monkeypatch.chdir(tmp_path);main();r=json.loads(Path('runs/reverse-morphology-residual-20260920.json').read_text())
 assert r['stats']['states']==256 and r['stats']['reverse_transitions']==256
 assert r['rejected_reverse_traces'] and r['independent_audit']['reverse_sha256']
 assert all(x['audit']['two_pointer_checked'] for x in r['complete_prose_controls'])
def test_exact(): assert audit('Able was I ere I saw Elba')['exact']
