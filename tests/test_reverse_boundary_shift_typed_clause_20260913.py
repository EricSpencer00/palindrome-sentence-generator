from experiments.reverse_boundary_shift_typed_clause_20260913 import run
def test_generated_typed_inventory_and_boundary_shift():
 r=run();assert r['config']['source_count']>=8;assert r['config']['boundary_shifting'];assert all(x['source_parse'] for x in r['source_records'])
def test_only_exact_admitted_candidates_are_exposed():
 r=run();assert all(x['independent_exact_audit']['exact'] and x['mechanically_admitted'] for x in r['admitted_candidates']);assert r['reader_status'].startswith('unreviewed')
