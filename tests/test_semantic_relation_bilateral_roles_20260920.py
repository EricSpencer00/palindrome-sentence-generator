import json,subprocess,sys
from pathlib import Path
def test_relation_lane():
 subprocess.run([sys.executable,'experiments/semantic_relation_bilateral_roles_20260920.py'],check=True)
 d=json.loads(Path('runs/semantic-relation-bilateral-roles-20260920.json').read_text())
 assert d['counts']['role_states']==12 and d['counts']['live_equations']>0
 assert d['counts']['exact_over_38']==0
 assert all('pointer_audit' in c['audit'] and 'sha256' in c['audit'] for c in d['controls'])
