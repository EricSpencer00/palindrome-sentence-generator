from experiments.dual_parse_csp_20260922 import LEFT, RIGHT, solve


def test_csp_rejects_short_recovery_and_keeps_bounded_frontier():
    result = solve(LEFT, RIGHT)
    assert result["results"] == []
    assert result["admission_rejections"] == 1
    assert result["operator"] == "dual-parse-csp-operator-20260922"


def test_operator_requires_exact_character_closure():
    result = solve(LEFT, RIGHT)
    assert result["transitions"] > 0
    assert all(row["exact_half_equation"] for row in result["results"])
