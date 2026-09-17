from experiments.fresh_seam_heldout_joint_repair_20260916 import run
def test_single_heldout_repair_is_not_a_sweep():
 p=run();assert p['stats']['heldout_repairs']==1;assert p['novelty_preflight']['status']=='passed';r=p['candidate'];assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['repair_operator']['held_out']
