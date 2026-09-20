from experiments.attachment_conditioned_reverse_parser_20260920 import audit,grammar,run,valid_attachment
def test_attachment_roles_are_typed():
 p,l,r,s=grammar(); assert any('PP_OBJ' in x for x in r) and any('REL_OBJ' in x for x in r) and any('RECIP' in x for x in r); assert valid_attachment(('NP','V','OBJ','PP_OBJ'))
def test_parser_runs_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['states']>0
 for row in x['rendered_candidates'][:20]: assert audit(row['rendered'])==row['audit']
