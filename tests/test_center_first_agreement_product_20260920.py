import json
from pathlib import Path
from experiments.center_first_agreement_product_20260920 import audit,main
def test_center_first_run(monkeypatch,tmp_path):
 monkeypatch.chdir(tmp_path); main(); r=json.loads(Path('runs/center-first-agreement-product-20260920.json').read_text())
 assert r['stats']['states']==64 and r['stats']['admitted']==6
 assert r['independent_audit']['forward_sha256'] and r['independent_audit']['reverse_sha256']
 assert all(x['audit']['two_pointer_checked'] for x in r['candidates']+r['complete_prose_controls'])
def test_known_exact_audit(): assert audit('Able was I ere I saw Elba')['exact']
