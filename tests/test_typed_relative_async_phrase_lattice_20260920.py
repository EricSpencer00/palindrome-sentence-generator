from experiments.typed_relative_async_phrase_lattice_20260920 import audit, run
def test_audit_and_relative_controls():
    assert audit("A man, a plan, a canal: Panama!")["pointer_exact"]
    x=run(); assert x["stats"]["rendered_controls"]==9; assert x["stats"]["online_states"]>0
    assert all(r["provenance"]["typed_relative_clause"] for r in x["controls"])
