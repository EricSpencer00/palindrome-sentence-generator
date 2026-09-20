from experiments.authored_crossword_seam_phrase_design_20260920 import audit,run
def test_crossword_seam_design():
 x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['states']>0;assert x['stats']['controls']>=20
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
