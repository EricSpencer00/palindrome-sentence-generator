from experiments.incumbent_498_event_frame_seam_repair_20261002 import build_payload


def test_varied_event_frame_pairs_are_exact_before_composition():
    payload = build_payload()
    assert all(row["pair_exact"] for row in payload["pair_library"].values())
    assert {"delivers", "stops", "maps", "deliver", "draw"} <= {
        predicate
        for row in payload["pair_library"].values()
        for predicate in row["left_predicates"]
    }


def test_every_repaired_child_is_exact_and_above_the_530_control():
    payload = build_payload()
    assert payload["stats"] == {
        "authored_paths": 5,
        "independently_exact_children": 5,
        "children_over_530": 5,
        "shortest_letters": 536,
        "longest_letters": 556,
        "maximum_inherited_outer_letters_replaced": 78,
    }
    for row in payload["rows"]:
        assert row["audit"]["letters"] > 530
        assert row["audit"]["two_pointer_exact"]
        assert row["audit"]["project_validator_exact"]
        assert row["audit"]["sha_equal"]


def test_frame_repair_reduces_saw_was_and_adds_predicates():
    for row in build_payload()["rows"]:
        repair = row["frame_repair"]
        assert repair["repaired_saw_was_count"] < repair["baseline_saw_was_count"]
        assert len(repair["repaired_outer_predicates"]) > len(repair["baseline_outer_predicates"])


def test_cleaner_554_and_longest_556_hashes_are_stable():
    rows = {row["id"]: row for row in build_payload()["rows"]}
    assert rows["depth39-varied-f1g1h1k"]["audit"]["sha256_forward"] == (
        "02e1ded5e201a2dac2b60a23c30eea7853527cab4ed99fcdb131ad2fd4aed08c"
    )
    assert rows["depth39-longest-f1g1h1r"]["audit"]["sha256_forward"] == (
        "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14"
    )
