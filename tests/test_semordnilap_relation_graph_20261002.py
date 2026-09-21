from experiments.semordnilap_relation_graph_20261002 import audit, observations, run, tape


def test_independent_audit_detects_exact_tape():
    result = audit("A man, a plan, a canal, Panama")
    assert result["two_pointer_exact"] is True
    assert result["sha256_forward"] == result["sha256_reverse"]


def test_relation_graph_outputs_are_exact_and_gated():
    rows = run(5)
    assert rows == []
    controls = observations(5)
    assert controls
    assert all("rendered" in row and "audit" in row for row in controls)


def test_tape_normalization():
    assert tape("A, B!") == "ab"
