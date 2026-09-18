from experiments.compositional_grammar_spine_20260918 import run, independent_audit

def test_lane_has_scene_first_grammar_and_independent_audit():
    p = run()
    assert p["construction"]["scene_spine_first"]
    assert p["construction"]["valency_automaton"]
    assert p["stats"]["fresh_nodes"] > 0
    for c in p["fresh_exact_closures"]:
        assert c["audit"]["exact"] and c["independent_audit"]["independent_exact"]

def test_controls_are_fresh_and_not_certified():
    p = run()
    assert p["rendered_candidates"]
    assert all(c["reader_status"] == "human-unreviewed" for c in p["rendered_candidates"])
    assert all(independent_audit(c["rendered"])["letters"] == c["audit"]["letters"] for c in p["rendered_candidates"])
