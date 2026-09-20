from experiments.semantic_selection_reverse_parser_20260920 import audit,grammar,run,selection_valid
def test_selection_constraints_are_typed():
 p,l,r,s=grammar(); assert any(x.animacy=='animate' for x in p['RECIP']); assert selection_valid(('NP','V','RECIP','OBJ'),[p['NP'][0],p['V'][0],p['RECIP'][0],p['OBJ'][0]])
def test_parser_runs_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['states']>0
 for row in x['rendered_candidates'][:20]: assert audit(row['rendered'])==row['audit']
