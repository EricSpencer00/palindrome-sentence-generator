from experiments.semantic_phrase_trie_orbit_20260920 import audit,banks,run
def test_phrase_trie_banks_and_name_endpoints():
 b=banks(); assert len(b['NP'])>=5 and len(b['VP'])>=5 and len(b['NAME'])>=5
def test_controls_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['seeds']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
