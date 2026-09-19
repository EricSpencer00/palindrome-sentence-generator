import json
from experiments.phrase_boundary_valency_equations_20260918 import audit, solve


def test_independent_audit_detects_nonexact_control(tmp_path, monkeypatch):
    d = audit("The patient curator records the brass compass before dawn.")
    assert d["letters"] == 49
    assert d["two_pointer_exact"] is False
    assert d["forward_sha256"] != d["reverse_sha256"]


def test_solver_records_pre_render_equation_pruning():
    d = solve()
    assert d["states"] == 1_562_500
    assert d["equation_pruned"] == d["states"]
    assert d["exact_count"] == 0
    assert d["strict_gate"]["admitted"] == 0
    assert d["intact_controls"]
