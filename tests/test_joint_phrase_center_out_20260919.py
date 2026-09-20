from experiments.joint_phrase_center_out_20260919 import independent_audit, search


def test_independent_audit_and_provenance():
    assert independent_audit("A man, a plan, a canal: Panama!")["exact"]
    data = search(limit=4)
    assert data["run_id"] == "joint-phrase-center-out-20260919"
    assert data["provenance"]["generated_compositionally"]
    assert data["candidates"]
    assert all("audit" in x and "provenance" in x for x in data["candidates"])
    assert sum(x["audit"]["exact"] for x in data["candidates"]) == data["stats"]["exact"]
