from pathlib import Path

from paper.build_readability_table import DEFAULT_INPUT, build_table


def test_long_output_table_is_exact_and_matches_controls():
    table = build_table(DEFAULT_INPUT)
    assert table.count("&") == 10 * 5 + 5
    assert "498 overhang & 498 & 161 & 0.582 & 1.630 & 1.048" in table
    assert "752 lineage end & 752 & 187 & 0.209 & 1.103 & 0.894" in table
    assert "Mean & --- & --- & 0.256 & 1.462 & 1.206" in table
    assert "not greater readability" in table
