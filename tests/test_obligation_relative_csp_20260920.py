from experiments.obligation_relative_csp_20260920 import audit, bank, index, run

def test_audit_independent_exact_seed():
    a = audit('An aide rips nine memos; some men inspire Diana.')
    assert a['exact'] and a['letters'] == 38 and a['sha_equal']

def test_relative_inventory_is_typed_and_indexed():
    inv = bank(); ix = index(inv)
    assert {'SUBJ','OBJ','REL','RVERB'} <= set(inv)
    assert sum(sum(len(bucket) for bucket in role.values()) for role in ix.values()) == sum(map(len, inv.values()))

def test_run_is_fail_closed_and_controls_audited():
    out = run()
    assert out['stats']['fresh_exact_gt38'] == len(out['exact_candidates']) == 0
    assert all('audit' in x and x['provenance']['generated'] is False for x in out['controls'])
