from experiments.reversible_predicate_center_extension_20260918 import run


def test_targeted_extension_reaches_longer_exact_diagnostic_but_not_admission():
    result = run()
    exact = [row for row in result["rendered_candidates"] if row["audit"]["two_pointer_exact"]]
    assert result["stats"]["longest_exact_letters"] == 54
    assert result["stats"]["mechanically_admitted"] == 0
    assert result["stats"]["reader_eligible"] == 0
    assert exact[0]["rendered"] == (
        "Was Noel an era, a gas, an item raw? War met in a, saga, arena, Leon saw."
    )
    assert any(row["audit"]["letters"] == 54 and row["pair"]["left"] == "smart"
               for row in exact)


def test_extension_keeps_hidden_span_evidence():
    result = run()
    exact = next(row for row in result["rendered_candidates"] if row["audit"]["two_pointer_exact"])
    assert not exact["mechanical_checks"]["no_self_palindromic_proper_multiword_span"]
    assert any(span["words"] == ["raw", "war"] for span in exact["hidden_palindromic_spans"])
