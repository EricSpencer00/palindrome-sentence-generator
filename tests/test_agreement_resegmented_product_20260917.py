import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("agreement_lane", Path(__file__).parents[1] / "experiments/agreement_resegmented_product_20260917.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
run = module.run

def test_agreement_product_is_long_and_rejects_controls():
    out=run()
    assert out["status"] == "completed_no_exact_closure"
    assert out["stats"]["exact"] == 0
    assert len(out["rows"]) == 3
    for row in out["rows"]:
        assert row["audit"]["letters"] >= 39
        assert row["audit"]["sha256"] != row["audit"]["reverse_sha256"]
        assert row["provenance"]["left_clause_emitted_forward"]
    assert all("fragment" in x["reason"] or "control" in x["reason"] for x in out["rejected_controls"])
