from experiments.center_terminal_clause_family_20260916 import run
def test_center_terminal_family_is_bounded_and_non_nested():
 p=run();assert p['stats']['bounded_states']==8;assert p['novelty_preflight']['status']=='passed'
 for r in p['candidates']:
  assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['nested_palindrome_span'] is False
