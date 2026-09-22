import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]

def test_residual_directed_abba_probe_is_reproducible_and_audited():
    p = subprocess.run([sys.executable, str(ROOT / 'experiments/typed_abba_residual_authoring_20260930.py')],
                       check=True, capture_output=True, text=True)
    out = json.loads(p.stdout)
    assert out['stats']['rendered_candidates'] == 5
    assert out['stats']['exact'] == 0
    assert out['provenance']['b2_authored_after_live_prefix']
    assert out['provenance']['a2_authored_against_residual']
    assert all('rendered' in row and 'audit' in row for row in out['rendered_candidates'])
    assert all(row['audit']['two_pointer_exact'] is False for row in out['rendered_candidates'])
    assert all(row['audit']['forward_sha256'] != row['audit']['reverse_sha256'] for row in out['rendered_candidates'])
