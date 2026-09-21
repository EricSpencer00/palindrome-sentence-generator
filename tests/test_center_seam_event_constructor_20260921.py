import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "experiments/center_seam_event_constructor_20260921.py"
spec = importlib.util.spec_from_file_location("center_seam", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_audit_independent_pointer_and_sha():
    a = m.audit("An aide rips nine memos; some men inspire Diana.")
    assert a["exact"] and a["letters"] == 38
    assert a["sha_forward"] == a["sha_reverse"]

def test_constructor_has_independent_typed_arcs():
    out = m.search()
    assert out["clauses"] > 0
    assert out["method"] == "center_seam_event_constructor"
    assert all(x["provenance"] == "independent_typed_event_arcs" for x in out["controls"])

def test_controls_are_not_claimed_readable():
    out = m.search()
    assert all(not x["reader_worthy"] for x in out["controls"])
