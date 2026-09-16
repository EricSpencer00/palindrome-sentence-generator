import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).parents[1]
RUN=ROOT/'runs/centerout-typed-semantic-debt-20260916.json'
def test_run_contract_and_independent_replay():
    subprocess.run([sys.executable,str(ROOT/'experiments/centerout_typed_semantic_debt_20260916.py')],check=True)
    d=json.loads(RUN.read_text()); assert d['novelty_preflight']['passed']; assert d['summary']['max_length']>100
    for r in d['rows']:
        t=''.join(c for c in r['rendered'].lower() if c.isalpha())
        assert r['audit']['length']==len(t)
        assert r['audit']['sha256_forward']==__import__('hashlib').sha256(t.encode()).hexdigest()
        assert r['audit']['sha256_reverse']==__import__('hashlib').sha256(t[::-1].encode()).hexdigest()
        assert r['provenance']['reversed_finished_sentence'] is False
        assert r['next_repair']
