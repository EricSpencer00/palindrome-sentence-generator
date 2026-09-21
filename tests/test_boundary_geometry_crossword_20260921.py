from experiments import boundary_geometry_crossword_20260921 as model


def test_boundary_constraint_keeps_withheld_benchmark_and_rejects_nested_regression():
    # Regression-only geometry: no benchmark words enter the constructor.
    benchmark = model.Geometry((2, 4, 4, 4, 5, 4, 3, 7, 5))
    rejected = model.Geometry((4, 7, 4, 1, 4, 7, 4))
    assert not benchmark.reflected_boundary_pairs
    assert rejected.reflected_boundary_pairs == ((4, 27), (11, 20), (15, 16))


def test_frozen_geometry_lengths_and_domains_are_supported():
    graphs = model.frozen_geometries()
    assert [g.length for g in graphs] == [41, 41, 43, 43, 47, 47]
    assert all(not g.reflected_boundary_pairs for g in graphs)
    assert all(g.connected for g in graphs)
    assert all(all(model.initial_domains(g)) for g in graphs)


def test_overlap_equations_recover_an_odd_center_inside_a_word(monkeypatch):
    monkeypatch.setattr(model, "ROLES", ("left", "right"))
    monkeypatch.setattr(model, "BANK", {"left": ["ab", "ac"], "right": ["cba", "dba"]})
    graph = model.Geometry((2, 3))
    result = model.solve(graph)
    assert {row["audit"]["normalized"] for row in result["solutions"]} == {"abcba", "abdba"}
    assert all(row["audit"]["sha256_forward"] == row["audit"]["sha256_reverse"] for row in result["solutions"])


def test_arc_consistency_explains_an_empty_domain(monkeypatch):
    monkeypatch.setattr(model, "ROLES", ("left", "right"))
    monkeypatch.setattr(model, "BANK", {"left": ["ab"], "right": ["cda"]})
    result = model.solve(model.Geometry((2, 3)))
    assert result["solutions"] == []
    assert result["first_obstruction"]["removed_words"]
    assert result["first_obstruction"]["overlap_equations"]


def test_pointer_audit_rejects_nonpalindrome_and_accepts_both_parities():
    for text in ("abcba", "abba"):
        result = model.audit(text)
        assert result["pointer_exact"]
        assert result["sha256_forward"] == result["sha256_reverse"]
    assert not model.audit("abdcba")["pointer_exact"]


def test_reflected_word_blocks_are_not_connected():
    assert not model.Geometry((3, 3, 3)).connected
