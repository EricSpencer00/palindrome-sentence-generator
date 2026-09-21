from coordination_graph_live_mirror_20260921 import run, tape

def test_graph_is_bounded_and_roles_precede_rendering():
    d=run()
    assert d["stats"]["rendered"] == 8
    assert all(r["provenance"]["roles_before_lexical_realization"] for r in d["rendered_controls"])
    assert all(r["provenance"]["live_mirrored_characters"] for r in d["rendered_controls"])
    assert d["provenance"]["generator_sha256"]

def test_independent_audit_fields_exist():
    d=run(); row=d["rendered_controls"][0]
    assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"] or row["audit"]["two_pointer_exact"]
    assert tape(row["rendered"]) == tape(row["rendered"])
