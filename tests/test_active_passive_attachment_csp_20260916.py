from experiments.active_passive_attachment_csp_20260916 import run
def test_active_passive_attachment_csp_is_fresh_and_audited():
 p=run();assert p['stats']['states']==8;assert p['novelty_preflight']['status']=='passed';assert not p['novelty_preflight']['dialogue_relative_templates_reused']
 for r in p['candidates']:
  assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['semordnilap_chain'] is False
