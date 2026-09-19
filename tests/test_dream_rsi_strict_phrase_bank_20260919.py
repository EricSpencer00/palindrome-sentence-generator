from experiments.dream_rsi_strict_phrase_bank_20260919 import strict_status


def test_hidden_span_is_a_hard_repair_gate():
    row = {
        "rendered": "Erased on forever event is an evening. Is sign in even as it never ever. Of nodes are.",
        "letters": 66,
        "audit": {
            "two_pointer_exact": True,
            "sha_equal_under_reversal": True,
        },
    }
    status = strict_status(row)
    assert status["exact_two_pointer_and_sha"]
    assert status["hidden_proper_span"]
    assert not status["mechanically_admitted"]
    assert status["rlaif_diagnostic"]["certifies_readability"] is False


def test_strict_status_does_not_promote_the_regression_anchor():
    row = {
        "rendered": "An aide rips nine memos; some men inspire Diana.",
        "letters": 38,
        "audit": {
            "two_pointer_exact": True,
            "sha_equal_under_reversal": True,
        },
    }
    assert not strict_status(row)["mechanically_admitted"]
