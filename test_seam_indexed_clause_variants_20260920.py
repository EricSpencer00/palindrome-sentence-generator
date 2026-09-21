import json
from seam_indexed_clause_variants_20260920 import run
def test_indexed_lane_has_compatible_pairs_and_full_audits():
 d=run(); assert d['stats']['compatible_pairs']>0; assert d['stats']['compatible_pairs']<d['stats']['clause_variants']**2
 assert len(d['diagnostic_controls'])==d['stats']['compatible_pairs']
 assert all('pointer_audit' in r and 'sha256_forward' in r['audit'] for r in d['diagnostic_controls'])
 assert d['stats']['exact_gt38']==0
