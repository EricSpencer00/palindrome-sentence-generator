from experiments.lexical_grammar_lm_intersection_20260920 import audit,consume,run
def test_live_phrase_crossing_and_audit():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['class_seeds']>0; assert x['stats']['states']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
def test_consumption_crosses_unequal_phrase_lengths():
 assert consume('an','diana')==('', 'dia'); assert consume('aide','inspiredia')==('', 'inspir')
