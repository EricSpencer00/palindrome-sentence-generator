from experiments.macrostep_frontier_rerank_20260913 import (
    endpoint_state, endpoint_witness_check, parse_selections, reopen_boundary,
)
from llm_palindrome.frontier_macros import state_from_anchors


def test_parse_selections_requires_anchor_preserving_complete_witness():
    state = endpoint_state("desserts", "stressed")
    menu = [{"id": "w", "state": state}]
    raw = '{"selections":[{"id":"w","witness":"Desserts to share and not stressed."}]}'
    selected = parse_selections(raw, menu, 4)
    assert selected == [{"id": "w", "witness": "Desserts to share and not stressed."}]

    bad = '{"selections":[{"id":"w","witness":"Desserts to share and happy."}]}'
    assert parse_selections(bad, menu, 4) == []


def test_reopen_boundary_only_emits_exactly_compatible_shortened_anchors():
    parent = state_from_anchors("desserts to", "not stressed")
    reopened = reopen_boundary(parent, "Desserts to share made the guests feel welcome and not stressed.")
    assert reopened
    assert all(row["reopened"]["left_drop"] + row["reopened"]["right_drop"] > 0
               for row in reopened)
    assert all(len(row["state"].left) <= len(parent.left)
               and len(row["state"].right) <= len(parent.right)
               for row in reopened)


def test_endpoint_witness_check_rejects_model_drift():
    assert endpoint_witness_check(("desserts", "stressed"),
                                  "Desserts pile up while the waiter feels stressed.")["valid"]
    assert not endpoint_witness_check(("deliver", "reviled"),
                                      "Deliver the package; the courier waits in the village.")["valid"]
