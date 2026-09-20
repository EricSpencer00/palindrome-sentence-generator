from experiments.lexical_reverse_segmentation_grammar_20260920 import audit,grammar,run
def test_complete_grammar_and_variable_trie():
 b,p,s=grammar(); assert len(s)>100; assert len(p)>=3; assert all(b[k] for k in ('NP','V','OBJ','PP','REL'))
def test_reverse_parser_and_controls_audit():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['states']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
