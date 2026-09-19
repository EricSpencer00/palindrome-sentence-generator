from experiments.authored_grammatical_clause_pairs_20260920 import run

def test_fresh_clause_lattice_audits_every_closure():
    result=run()
    assert result["stats"] == {"nodes":49,"exact":49,"longest_exact_letters":30,"mechanically_admitted":0}
    assert all(x["audit"]["two_pointer_exact"] for x in result["candidates"])
    assert all(x["audit"]["sha256_forward"]==x["audit"]["sha256_reverse"] for x in result["candidates"])
