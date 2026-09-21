import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("world_probe", Path(__file__).parents[1] /
    "experiments/world_preimage_character_product_20260921.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_world_constraints_change_accepted_language_before_rendering():
    live, ablated = probe.Graph(True), probe.Graph(False)
    valid = "An aide opens the gate. The porter delivers a parcel."
    impossible = "An aide rips nine memos. Some men read nine memos."
    assert live.accepts(valid) and ablated.accepts(valid)
    assert not live.accepts(impossible) and ablated.accepts(impossible)


def test_character_intersection_has_independent_exact_forward_paths():
    graph = probe.Graph()
    result = graph.solve()
    assert result["status"] == "exhausted"
    assert result["witnesses"]
    for text in result["witnesses"]:
        assert graph.accepts(text)
        a = probe.audit(text)
        assert a["pointer_exact"]
        assert a["sha256_forward"] == a["sha256_reverse"]


def test_resource_deletion_cannot_be_undone_without_an_action():
    assert probe.replay(["open", "deliver"])["valid"]
    assert not probe.replay(["open", "close", "deliver"])["valid"]
    assert not probe.replay(["rip", "read"])["valid"]


def test_shortcut_audit_catches_nested_and_repeated_units():
    a = probe.audit("Some men. Some men. Never odd or even.")
    assert a["repeated_sentences"]
    assert "never odd or even" in a["proper_self_palindromic_spans"]
