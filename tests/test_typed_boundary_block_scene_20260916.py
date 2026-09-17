from experiments.typed_boundary_block_scene_20260916 import run
def test_typed_blocks_are_connected_and_not_old_chain():
 p=run();assert p['stats']['bounded_assignments']==16;assert p['novelty_preflight']['status']=='passed';assert p['reference_control']['old_command_chain_used'] is False
 for r in p['candidates']:
  assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['word_order_symmetry'] is False
