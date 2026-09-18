from experiments.character_crossing_relative_clause import audit, letters
from experiments.semantic_slot_solver import solve


def test_independent_exact_solver_admits_a_known_control_tape():
    domains = [["live"], ["on"], ["time"], ["emit"], ["no"], ["evil"]]
    found, stats = solve(domains, min_letters=20, max_letters=20)
    assert found == ["live on time emit no evil"]
    assert stats["closed_exact"] == 1


def test_audit_enforces_every_hard_mechanical_gate():
    text = "Live on time, emit no evil."
    tape = letters(text)
    checks = audit(text, {tape})
    assert checks["exact_letter_palindrome"] is True
    assert checks["at_least_30_letters"] is False
    assert checks["local_catalogue_absent"] is False
    assert checks["not_word_order_symmetry"] is False


def test_relative_clause_final_gate_rejects_catalogue_family_surface():
    checks = audit("Marge lets Hara see Sarah's telegram.", set())
    assert checks["not_catalogue_family_derivative"] is False


def test_near_miss_is_not_admitted():
    checks = audit("Rachel kept the letter that Daniel sent.", set())
    assert checks["exact_letter_palindrome"] is False
