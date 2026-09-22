from experiments.dependency_phrase_path_lattice_20260930 import bank_checks, run
from llm_palindrome.validator import is_palindrome

def test_dependency_bank_has_typed_roles_and_no_authored_reversals():
    checks = bank_checks()
    assert checks["reverse_pair_free"]
    assert checks["dependency_order"][:3] == ["NP:subject", "VP:finite", "NP:object"]

def test_every_reported_exact_is_independently_audited():
    result = run()
    assert result["solver_stats"]["cap_reached"] is False
    for row in result["exact_candidates"]:
        assert row["audit"]["two_pointer_exact"] and row["audit"]["sha_equal"]
        assert is_palindrome(row["rendered"])

def test_controls_include_intact_prose_and_incumbent():
    result = run()
    assert any(c["kind"] == "intact_prose_control" for c in result["controls"])
    assert any(c["kind"] == "incumbent_exact_control" and c["audit"]["two_pointer_exact"]
               for c in result["controls"])
