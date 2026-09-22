from experiments.mixed_clause_live_intersection_20260930 import run, independent_validator


def test_mixed_clause_run_is_a_live_search_and_has_no_new_closure():
    out = run()
    assert out["result"]["states"] >= 1
    assert out["result"]["transitions"] == 0
    assert out["rendered_candidates"] == []
    assert out["positive_control"]["validator"]


def test_independent_validator_and_provenance():
    out = run()
    assert independent_validator(out["positive_control"]["rendered"])
    p = out["provenance"]
    assert all(not p[k] for k in ("complete_sentence_enumeration", "catalogue_text",
                                  "reversed_phrase_bank", "repeated_self_palindromic_units",
                                  "post_hoc_repair", "per_candidate_rlaif"))
