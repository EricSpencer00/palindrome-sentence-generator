from experiments.twochar_outer_family_20260920 import run
def test_twochar_family_is_filtered_and_audited():
 d=run(); assert d['stats']['possible_pairs']>d['stats']['compatible_pairs']>0
 assert len(d['diagnostic_controls'])==d['stats']['compatible_pairs']
 assert all(r['pointer_audit']['independent_exact'] is False for r in d['diagnostic_controls'])
