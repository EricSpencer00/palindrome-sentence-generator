from experiments.synchronous_cfg_intersection_20260919 import consume, audit

def test_consume_retains_unequal_word_residuals():
    assert consume("abcd", "cba") == ("d", "")
    assert consume("abc", "xycba") == ("", "xy")
    assert consume("d", "d") == ("", "")
    assert consume("ab", "cd") is None

def test_independent_audit_rejects_nonpalindrome():
    assert audit("An aide rips nine memos; some men inspire Diana.") ["exact"] is True
    assert audit("a grammatical sentence") ["exact"] is False
