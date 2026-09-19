from experiments.live_clause_pair_dfs_20260920 import consume, search


def test_live_consumer_carries_and_closes_obligation():
    debt, side = consume("an", "anaid", "", "")
    assert (debt, side) == ("aid", "R")
    debt, side = consume("aide", "", debt, side)
    assert (debt, side) == ("e", "L")
    debt, side = consume("", "eripsni", debt, side)
    assert (debt, side) == ("ripsni", "R")


def test_short_calibration_is_independently_exact_and_long_form_is_gated():
    short = search(min_letters=38, long_form=False)
    assert short["stats"]["exact"] >= 1
    candidate = short["rendered_candidates"][0]
    assert candidate["audit"]["two_pointer_exact"]
    assert candidate["audit"]["sha256_forward"] == candidate["audit"]["sha256_reverse"]
    long_form = search(min_letters=39, long_form=True)
    assert long_form["stats"]["exact"] == 0
