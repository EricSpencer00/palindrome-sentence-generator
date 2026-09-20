from experiments.dependency_role_permutation_chart_20260920 import audit,run
def test_dependency_chart():
 x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['chart_pairs']>0
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
