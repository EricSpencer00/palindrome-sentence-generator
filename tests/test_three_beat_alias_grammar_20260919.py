from experiments.three_beat_alias_grammar_20260919 import audit, run

def test_independent_seed_audit():
 a=audit("An aide rips nine memos; some men inspire Diana.")
 assert a["two_pointer_exact"] and a["sha_equal"]

def test_three_distinct_beats_and_frontier_controls():
 r=run()
 assert r["stats"]["rendered"] > 0
 assert r["stats"]["frontier_controls"] > 0
 assert all(len(x["provenance"]["beats"]) == 3 for x in r["actual_candidates"])
 assert all(x["provenance"]["rlaif_used"] is False for x in r["actual_candidates"])
