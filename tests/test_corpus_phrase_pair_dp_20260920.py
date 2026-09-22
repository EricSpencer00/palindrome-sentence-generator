from experiments.corpus_phrase_pair_dp_20260920 import audit, run, letters, outer_score

def test_dp_generates_two_forward_prose_sides_and_audits():
    result = run()
    assert result["stats"]["transitions"] > 0
    assert result["stats"]["rendered_candidates"] > 0
    assert all(r["complete_prose"] and r["provenance"]["forward_generated_both_sides"]
               for r in result["rendered_candidates"])
    assert all(len(r["audit"]["sha256_forward"]) == 64 for r in result["rendered_candidates"])

def test_outer_score_is_not_a_palindrome_certificate():
    assert outer_score("the patient teacher", "the evening witness") >= 0
    assert not audit("the patient teacher, and the evening witness.")["exact"]

def test_no_finished_tape_shortcut():
    result = run()
    assert all(not r["provenance"]["finished_tape_reversal"] and
               not r["provenance"]["post_hoc_repair"]
               for r in result["rendered_candidates"])
