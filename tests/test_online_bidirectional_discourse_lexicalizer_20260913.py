from experiments.online_bidirectional_discourse_lexicalizer_20260913 import replay,run
def test_role_index_precedes_broad_source_inventory():
 r=run();assert len(r['source_inventory'])>8;assert r['config']['source_terminal_role_derived_before_expansion'];assert r['config']['fixed_terminal_word'] is False
def test_diagnostics_require_nonvacuous_thresholds():
 r=run();assert all(x['replay']['completed_target_words']>=2 and x['replay']['cross_source_boundary_transitions']>=1 for x in r['diagnostics']);assert r['config']['search_status']=='exhausted'
def test_tampered_ledger_is_rejected_and_exact_survivors_are_gated():
 r=run();
 for x in r['diagnostics']:
  if x['ledger']:
   bad=[dict(e) for e in x['ledger']];bad[0]['char']='z';assert not replay(bad)['ok'];break
 assert all(x['independent_exact_audit']['exact'] and x['mechanically_admitted'] for x in r['admitted_exact_survivors'])
