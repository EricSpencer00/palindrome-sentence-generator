from experiments.independent_unequal_clause_scheduler_20260920 import audit,clause_paths,consume,run
def test_paths_have_unequal_phrase_counts():
 ps=clause_paths(); assert len({len(x.words) for x in ps})>=3
def test_scheduler_runs_and_controls_audit():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['seeded']>0; assert x['stats']['states']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
 assert consume('aide','inspiredia')==('', 'inspir')
