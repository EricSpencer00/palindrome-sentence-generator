from experiments.broad_name_bank_complete_sentence_generator_20260920 import audit,run
def test_broad_bank_generator():
 x=run(limit=2000);assert x['novelty_preflight']['status']=='passed';assert x['stats']['pair_states']>0
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
