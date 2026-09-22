from experiments.two_sided_unmatched_buffer_dp_20260920 import consume
def test_equal_synthetic_fixture_passes():
 assert consume('ab','ab') == ('','')
 assert consume('ab','ba') is None
def test_unequal_buffer_carries_remainder():
 assert consume('ab','a') == ('b','')
def test_mismatch_fixture_prunes():
 assert consume('ax','ab') is None
