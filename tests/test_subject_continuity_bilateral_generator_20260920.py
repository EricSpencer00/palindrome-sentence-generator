import json,subprocess,sys
from pathlib import Path
def test_bilateral_run():
 subprocess.run([sys.executable,'experiments/subject_continuity_bilateral_generator_20260920.py'],check=True)
 d=json.loads(Path('runs/subject-continuity-bilateral-generator-20260920.json').read_text())
 assert d['counts']['states']==36 and d['counts']['live_equations']>0
 assert d['counts']['exact_over_38']==0
 assert all('pointer_audit' in c['audit'] and 'sha256' in c['audit'] for c in d['controls'])
