from experiments.dual_plan_residual_search_20260914 import (
    PLANS, ascii_audit, cancel, search_pair,
)


def test_cancel_crosses_staggered_word_boundaries():
    assert cancel("anaide", "ana", 1) == ("ide", 1)
    assert cancel("ide", "iderips", 1) == ("rips", -1)
    assert cancel("rips", "rips", -1) == ("", 0)


def test_seed_plan_is_recovered_by_live_residual_search():
    left = next(plan for plan in PLANS if plan.name == "numbered_svo")
    right = next(plan for plan in PLANS if plan.name == "plural_name")
    rows, stats = search_pair(left, right, state_budget=250_000)
    texts = {row["text"].lower() for row in rows}
    assert "an aide rips nine memos; some men inspire diana." in texts
    assert stats["states"] > 0


def test_independent_ascii_audit():
    assert ascii_audit("Drawer reward.")["exact"]
    assert not ascii_audit("ordinary prose")["exact"]
