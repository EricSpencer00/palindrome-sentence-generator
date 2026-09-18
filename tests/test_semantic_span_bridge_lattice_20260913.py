from experiments.semantic_span_bridge_lattice_20260913 import run
def test_broad_typed_inventory_and_independent_witnesses():
 r=run();assert r['config']['source_count']>8;assert all(x['source_parse'] for x in r['source_records']);assert r['config']['cross_source_word_boundaries']
def test_zero_result_reports_status_and_dead_frontiers():
 r=run();assert r['config']['search_status'] in {'exhausted','truncated'};assert r['exact_survivors']==[];assert r['admitted_exact_survivors']==[];assert r['dead_frontiers']
def test_any_exact_survivor_is_fully_witnessed_and_admitted():
 r=run();assert all(x['independent_exact_audit']['exact'] and x['independent_source_parse'] and x['independent_right_parse'] and x['mechanically_admitted'] for x in r['admitted_exact_survivors'])
