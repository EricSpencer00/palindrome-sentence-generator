import json
from pathlib import Path
from experiments.live_adjunct_residual_lattice_20260920 import audit,main
def test_lattice(monkeypatch,tmp_path):
 monkeypatch.chdir(tmp_path);main();r=json.loads(Path('runs/live-adjunct-residual-lattice-20260920.json').read_text())
 assert r['stats']['states']==256 and r['stats']['live_transitions']==256
 assert r['stats']['admitted'] >= 0 and r['independent_audit']['forward_sha256']
 assert all(x['audit']['two_pointer_checked'] for x in r['candidates']+r['complete_prose_controls'])
def test_exact(): assert audit('Able was I ere I saw Elba')['exact']
