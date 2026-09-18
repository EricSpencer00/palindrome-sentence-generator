from experiments.seed_centered_context_growth_20260918 import run, independent_audit


def test_context_lane_has_live_equations_and_withheld_seed():
    p = run()
    assert p["construction"]["live_character_equations"]
    assert p["withheld_benchmark"]["used_as_output"] is False
    assert p["novelty_preflight"]["duplicate_sweep"] is False
    assert p["search"]["stats"]["fresh_exact"] == 0


def test_independent_audit_agrees_on_palindrome_and_nonpalindrome():
    assert independent_audit("A man, a plan, a canal: Panama") ["two_pointer_exact"]
    assert not independent_audit("fresh prose") ["two_pointer_exact"]
