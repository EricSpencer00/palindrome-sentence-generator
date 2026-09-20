import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).parents[1]
def test_cross_role_csp_artifact_and_audits():
    subprocess.run([sys.executable,str(ROOT/'experiments/cross_role_clause_csp_20260920.py')],check=True,cwd=ROOT)
    x=json.loads((ROOT/'runs/cross-role-clause-csp-20260920.json').read_text())
    assert x['stats']['shape_pairs']==25 and x['stats']['states']>0
    assert x['stats']['fresh_exact_gt38']==0
    assert x['controls'][0]['audit']['sha_equal'] is True
    assert all(c['audit']['sha_equal'] is False for c in x['controls'][1:])
    assert x['novelty_preflight']['finished_tape_reversal'] is False
