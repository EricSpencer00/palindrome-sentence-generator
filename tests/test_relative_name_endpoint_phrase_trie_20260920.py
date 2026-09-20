from experiments.relative_name_endpoint_phrase_trie_20260920 import audit,banks,run
def test_relative_units_and_name_endpoints():
 b=banks(); assert len(b['REL'])>=4 and any('Diana' in p.text for p in b['REL'])
def test_controls_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['seeds']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
