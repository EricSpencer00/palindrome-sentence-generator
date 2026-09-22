import json
from pathlib import Path

from experiments.typed_phrase_graph_sara_window_edit_20260930 import audit, main, tape


def test_sara_window_edit_is_exact_and_keeps_parent_outside_tape():
    main()
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "runs/typed-phrase-graph-sara-window-edit-20260930.json").read_text()
    )
    candidate = result["candidate"]
    assert candidate["left"] == "Sara, did I live?"
    assert candidate["right"] == "Evil, I did, Aras."
    assert candidate["window_tape_reverse"]
    assert candidate["outside_tape_preserved"]
    assert candidate["audit"]["letters"] == 238
    assert candidate["audit"]["two_pointer_exact"]
    assert candidate["audit"]["validator_exact"]
    assert candidate["audit"]["sha_equal"]


def test_audit_independently_rejects_a_broken_window():
    text = "Sara, did I live? Evil, I did, Arasx."
    result = audit(text)
    assert not result["two_pointer_exact"]
    assert not result["validator_exact"]
    assert not result["sha_equal"]


def test_variant_inventory_is_explicit():
    main()
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "runs/typed-phrase-graph-sara-window-edit-20260930.json").read_text()
    )
    assert len(result["variants"]) == 5
    assert all(row["outside_tape_preserved"] for row in result["variants"])
