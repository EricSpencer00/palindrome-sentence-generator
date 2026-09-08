from llm_palindrome.pareto import pareto_front


def test_pareto_front_keeps_tradeoffs_and_removes_dominated_rows():
    rows = [
        {"name": "fluent", "fluency": 3, "diversity": 1},
        {"name": "diverse", "fluency": 1, "diversity": 3},
        {"name": "weak", "fluency": 1, "diversity": 1},
    ]
    names = {row["name"] for row in pareto_front(rows, ("fluency", "diversity"))}
    assert names == {"fluent", "diverse"}
