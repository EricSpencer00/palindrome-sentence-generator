from experiments.boundary_seeded_asymmetric_clause_lattice_20260920 import audit,run
def test_boundary_seed_lattice():
 x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['endpoint_seed_pairs']>0;assert x['stats']['controls']>=20
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
