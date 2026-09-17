import json
from pathlib import Path
def test_role_frames():
 x=json.loads(Path('runs/cfg-midpoint-role-frames-heldout-20260917.json').read_text()); assert x['candidate_count']==576 and x['exact_count']==0
 assert x['heldout_count']>0
 for r in x['diagnostic_controls']: assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal'] and r['grammar_state']['complete_obligations']
def test_no_shortcuts(): assert json.loads(Path('runs/cfg-midpoint-role-frames-heldout-20260917.json').read_text())['admitted_renderings']==[]
