from pathlib import Path

from paper.build_readability_table import DEFAULT_INPUT, build_table


def test_all_candidate_scores_are_exact_and_match_heldout_controls():
    table = build_table(DEFAULT_INPUT)
    assert table.count("&") == 13 * 5 + 5
    assert "38 inherited example & 38 & 9 & 0.515 & 0.025 & -0.490" in table
    assert "54 NP candidate & 54 & 13 & 0.407 & 1.923 & 1.516" in table
    assert "498 overhang search & 498 & 161 & 0.582 & 1.590 & 1.008" in table
    assert "752 center path & 752 & 187 & -0.297 & 1.201 & 1.499" in table
    assert "Long set mean ($n=9$) & --- & --- & -0.109 & 1.522 & 1.631" in table
    assert "All cases mean ($n=11$) & --- & --- & -0.006 & 1.423 & 1.428" in table
    assert "not greater human readability" in table
