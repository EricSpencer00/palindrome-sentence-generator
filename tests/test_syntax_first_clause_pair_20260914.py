from experiments.syntax_first_clause_pair_20260914 import exact_audit, enumerate_clauses, run


def test_independent_audit_is_letter_level():
    assert exact_audit("A drawer; reward a.")["exact"]
    assert not exact_audit("A drawer; rewards a.")["exact"]


def test_enumeration_is_typed_and_nonempty():
    rows = enumerate_clauses()
    assert len(rows) > 100_000
    assert {row["plan"] for row in rows} == {"svo", "adj_svo", "adv_svo", "pp_svo"}
    assert all(len(row["words"]) == len(row["roles"]) for row in rows)


def test_run_reports_provenance_and_does_not_claim_readability():
    result = run()
    assert result["config"]["typed_clause_enumeration_before_tape_matching"]
    assert result["provenance"]["material"].startswith("authored")
    assert result["scope"].startswith("A zero-pair")
    assert all(row["reader_status"] == "human-unreviewed" for row in result["pairs"])
