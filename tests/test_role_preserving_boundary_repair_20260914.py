from experiments.role_preserving_boundary_repair_20260914 import repaired_rows, run
from experiments.syntax_first_clause_pair_20260914 import enumerate_clauses


def test_repairs_preserve_a_declared_role():
    rows = repaired_rows(enumerate_clauses()[:20], 20)
    assert rows
    assert all(row["repair"]["role"] in row["roles"] for row in rows)
    assert all(row["words"][row["repair"]["slot"]] == row["repair"]["replacement"] for row in rows)


def test_run_records_dead_end_and_next_operator():
    result = run(max_base=100, max_repairs=200, max_pairs=3)
    assert result["config"]["role_preserving"]
    assert result["residual_evidence"]["dead_before_full_tape"] > 0
    assert result["exact_closure_count_seen"] == len(result["rendered_candidates"])
    assert "two coordinated role repairs" in result["next_constructive_operator"]
