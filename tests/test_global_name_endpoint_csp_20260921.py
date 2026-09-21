from experiments.global_name_endpoint_csp_20260921 import LEXICON, independent_audit, run, solve

def test_typed_names_are_ordinary_and_search_is_bounded():
    names = LEXICON["name"]
    assert all(n.lower() != n.lower()[::-1] for n in names)
    assert {"Diana", "Adrian", "Leon", "Noel"} <= set(names)
    x = solve(limit=300)
    assert x["stats"]["nodes"] <= 300
    assert x["state_model"]["typed_name_endpoint"]
    for words in x["found"]:
        assert words[1] != words[7] != words[9]
        assert independent_audit(words)["exact"] == ("".join(words) == "".join(words)[::-1])

def test_contract_and_queue_row():
    x = run(limit=300)
    assert x["config"]["letter_band"] == [39, 60]
    assert x["novelty_preflight"]["status"] == "passed"
    assert x["independent_exact_shortcut_audits"]
    assert x["state_model"]["typed_name_object_endpoint"]
    assert x["queue_row"]["status"] == "bounded residual"
    assert all(r["provenance"]["ordinary_name_entries_only"] for r in x["records"])
