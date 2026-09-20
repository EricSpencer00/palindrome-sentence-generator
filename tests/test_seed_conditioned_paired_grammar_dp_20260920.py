from experiments.seed_conditioned_paired_grammar_dp_20260920 import audit,run
def test_seed_geometry_is_calibration_only():
 x=run(max_left=4000);assert x['novelty_preflight']['status']=='passed';assert x['baseline_control']['calibration_only'];assert x['baseline_control']['audit']['exact'];assert x['stats']['dp_states']>0
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
