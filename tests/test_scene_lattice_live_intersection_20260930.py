import json
from experiments.scene_lattice_live_intersection_20260930 import audit, run


def test_audit_independent_exactness():
    row = audit("Able was I ere I saw Elba")
    assert row["two_pointer_exact"]
    assert row["sha_equal"]


def test_scene_lattice_artifact_and_controls():
    payload = run()
    result = payload["results"][0]
    assert result["states"] > 0
    assert payload["controls"]
    assert all(c["audit"]["letters"] > 0 for c in payload["controls"])
    assert payload["provenance"]["reader_evidence"] is False
