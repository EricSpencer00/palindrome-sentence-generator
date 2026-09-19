from experiments.half_tape_relative_slots_z2_20260919 import TEMPLATES, search


def test_relative_repair_exposes_internal_slots_and_preserves_exact_gate():
    assert any({"R", "RV", "RO"}.issubset(template) for template in TEMPLATES)
    rows, nodes = search(TEMPLATES[0], 50, max_nodes=500)
    assert nodes > 0
    assert rows == []
