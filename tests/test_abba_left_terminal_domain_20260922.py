from experiments.abba_left_terminal_domain_20260922 import audit, run


def test_independent_audit_and_seed():
    assert audit("An aide rips nine memos; some men inspire Diana.")["two_pointer_exact"]
    assert not audit("A quiet sailor studies the harbor.")["two_pointer_exact"]


def test_terminal_domain_selects_red_and_retains_elc_certificate():
    data = run()
    assert data["stats"]["closed_derivations"] == 0
    assert data["stats"]["exact_gt38"] == 0
    branches = data["terminal_domain_branches"]
    assert {b["reverse_onset"] for b in branches} == {"red", "elc"}
    assert any(b["reverse_onset"] == "red" and b["right_subject_count"] == 2
               for b in branches)
    assert any(b["reverse_onset"] == "elc" and b["grammar_certificate"]
               for b in branches)


def test_all_controls_are_intact_nonexact_prose():
    data = run()
    assert len(data["controls"]) == 4
    assert all(c["audit"]["two_pointer_exact"] is False for c in data["controls"])
