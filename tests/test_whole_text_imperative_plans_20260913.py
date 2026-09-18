from experiments.whole_text_imperative_plans_20260913 import run
def test_separate_typed_imperative_plans_use_kernel():
 r=run();assert r['config']['plan_count']==4;assert r['config']['kernel_center_inside_word'];assert all(x['semantic_plan'] for x in r['plan_runs'])
def test_finite_scope_and_exact_records_are_audited():
 r=run();assert r['config']['search_status'] in {'exhausted','truncated'};assert all(x['independent_exact_audit']['exact'] and x['independent_plan_parse'] and x['mechanically_admitted'] for x in r['admitted_exact_survivors'])
def test_zero_output_does_not_claim_readability():
 r=run();assert r['reader_status'].startswith('unreviewed');assert r['stats']['states']>0
