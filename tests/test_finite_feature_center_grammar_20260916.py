import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.finite_feature_center_grammar_20260916 import run


def test_finite_feature_center_run_is_preflighted_and_independently_audited():
    payload = run()
    assert payload["novelty_preflight"]["status"] == "passed"
    assert payload["grammar"]["center_production"]["lhs"] == "CENTER"
    assert payload["stats"]["candidates"] == 3
    assert payload["stats"]["over_100_letters"] >= 1
    for row in payload["rendered_candidates"]:
        assert row["anti_shortcut"]["intact_multi_clause_prose"]
        assert not row["anti_shortcut"]["word_order_mirror"]
        assert row["audit"]["two_pointer"]["exact"] is False
        assert row["audit"]["sha_equal"] is False
        assert row["live_character_state"]["emitted_from_independent_arms"]
