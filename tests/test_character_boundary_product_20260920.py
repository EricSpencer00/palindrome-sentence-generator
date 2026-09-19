from experiments.character_boundary_product_20260920 import run

def test_product_is_bounded_audited_and_no_shortcuts():
    p = run()
    assert p["stats"]["bounded_assignments"] == 4
    assert p["novelty_preflight"]["status"] == "passed"
    for row in p["candidates"]:
        assert row["orbit_lock"]["locked_before_render"]
        assert row["provenance"]["independently_authored_clauses"]
        assert not row["provenance"]["finished_tape_reversal"]
        assert not row["provenance"]["word_order_mirror"]
        assert row["exact_audit"]["sha256_forward"] != row["exact_audit"]["sha256_reverse"]
