from clause_reverse_cfg_center_20260920 import run
def test_clause_cfg_is_zero_or_audited():
 x=run(); assert x['novelty_preflight']['status'] in {'passed','zero-frontier'}
 for row in x['rendered_candidates']:
  assert row['provenance']['exact_residual_buffers_before_render']
  assert row['audit']['exact']
