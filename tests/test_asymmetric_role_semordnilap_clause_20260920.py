from experiments.asymmetric_role_semordnilap_clause_20260920 import audit,run
def test_asymmetric_clause_lane():
    x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['controls']>=20;assert x['stats']['states']>0
    for row in x['controls']:assert audit(row['rendered'])==row['audit']
    assert all('writes the analyst' not in row['rendered'] for row in x['controls'])
