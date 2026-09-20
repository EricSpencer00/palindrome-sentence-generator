from experiments.lexical_reverse_independent_role_parser_20260920 import audit,grammar,run
def test_independent_right_role_paths_exist():
 p,paths,s=grammar(); assert len(paths)>=4 and 'RECIP' in paths[-1]; assert len(s)>100
def test_parser_audits_rendered_rows():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['states']>0
 for row in x['rendered_candidates'][:20]: assert audit(row['rendered'])==row['audit']
