from experiments.morphology_outer_seam_solver_20260918 import run
def test_lane_records_audited_renderings():
 p=run(); assert p['construction']['live_character_constraints_before_render']; assert p['construction']['independent_audit']=='two_pointer_and_sha256'; assert all('audit' in x for x in p['rendered_candidates']); assert p['novelty_preflight']['prior_lane_reused'] is False
