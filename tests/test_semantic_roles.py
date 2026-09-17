from llm_palindrome.semantic_roles import Clause, search, two_pointer_sha


def test_audit_is_independent_and_reports_mismatch():
    audit = two_pointer_sha("Quiet harbor")
    assert audit["exact"] is False
    assert audit["sha256"] != audit["sha256_reverse"]


def test_clause_search_is_bounded_and_fail_closed():
    result = search([Clause("scene", "Mara reads maps.")], max_clauses=1)
    assert result["status"] == "no_closure"
    assert result["near_miss"]["audit"]["exact"] is False
    assert result["bounds"] == {"clauses": 1, "max_clauses": 1}
