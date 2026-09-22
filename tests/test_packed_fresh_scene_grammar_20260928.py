from experiments.packed_fresh_scene_grammar_20260928 import bank_checks, run


def test_fresh_scene_bank_has_no_authored_reverse_phrase_pairs():
    checks = bank_checks()
    assert checks["reverse_pair_free"]
    assert checks["phrase_reverse_pairs"] == []


def test_packed_scene_search_is_bounded_and_audited():
    result = run()
    assert result["solver_stats"]["cap_reached"] is False
    for row in result["accepting_witnesses"]:
        assert row["audit"]["two_pointer_exact"]
        assert row["audit"]["sha_equal"]
