from experiments.grammar_product_automaton_intersection_20260920 import audit,paths,run
def test_complete_paths_exist_before_product():
 ps=paths(); assert len(ps)>100; assert len({len(x.words) for x in ps})>=3
def test_product_runs_without_boundary_seed_and_audits_controls():
 x=run(state_limit=12000); assert x['novelty_preflight']['outer_boundary_seed'] is False; assert x['stats']['states']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
