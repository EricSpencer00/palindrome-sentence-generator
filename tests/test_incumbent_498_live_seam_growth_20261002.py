from experiments.incumbent_498_live_seam_growth_20261002 import (
    PARENT_SHA256,
    build_payload,
    independent_tape,
    load_parent,
)


def test_mandated_498_parent_is_loaded_and_independently_verified():
    _, rendered, tape = load_parent()
    assert len(tape) == 498
    assert tape == tape[::-1]
    assert independent_tape(rendered) == tape
    assert PARENT_SHA256 == "e809a2a05a414347615f68f27f6b4974aa00ede287fdb2a4b23f6ffbadec9032"


def test_live_seam_records_the_partial_word_owner_residual_and_cursors():
    seam = build_payload()["live_seam"]
    assert seam["left_cursor"]["normalized_offset"] == 2
    assert seam["right_cursor"]["normalized_offset"] == 495
    assert seam["partial_word"] == "(et)|g in get"
    assert seam["boundary_owner"] == "R"
    assert seam["trace"][0]["residual_after"] == "g"
    assert seam["trace"][1]["residual_after"] == "pamate"
    assert seam["trace"][2]["residual_after"] == ""


def test_every_saved_child_is_exact_novel_content_and_longer_than_control():
    payload = build_payload()
    assert payload["stats"] == {
        "authored_paths": 10,
        "independently_exact_children": 10,
        "children_over_530": 10,
        "shortest_letters": 538,
        "longest_letters": 554,
    }
    for row in payload["rows"]:
        audit = row["audit"]
        assert audit["two_pointer_exact"]
        assert audit["project_validator_exact"]
        assert audit["sha_equal"]
        assert audit["letters"] > 530
        assert row["new_content_words"]
        assert not row["repetition_control"]


def test_primary_550_child_has_stable_independent_hash():
    rows = {row["id"]: row for row in build_payload()["rows"]}
    row = rows["depth10-live-seam-ab"]
    assert row["audit"]["letters"] == 550
    assert row["audit"]["sha256_forward"] == "923ea4a184d775bead1a8c5a66e16331dc8966dec156c44659c81fe11a5b2582"


def test_primary_live_g_frontier_exceeds_the_duplicate_control():
    rows = {row["id"]: row for row in build_payload()["rows"]}
    assert rows["depth2-live-g-ab"]["audit"]["sha256_forward"] == (
        "14e55f616284b264d3c97be42a4fcdfdf5651345732054dcc7cad80613290fcd"
    )
    assert rows["depth2-live-g-ac"]["audit"]["letters"] == 554
    assert rows["depth2-live-g-ac"]["audit"]["sha256_forward"] == (
        "cc7f98df008c250649a3c5bec7c1062f3d1349b539c97391c9e2ab961208ab42"
    )


def test_deeper_cursor_frontier_replaces_more_inherited_filler():
    payload = build_payload()
    seams = {seam["normalized_depth"]: seam for seam in payload["alternate_live_seams"]}
    assert seams[10]["partial_word"] == "m|e"
    assert seams[35]["partial_word"] == "op|en"
    assert seams[35]["retained_letters"] == 428
    assert seams[39]["partial_word"] == "i|ts"
    rows = {row["id"]: row for row in payload["rows"]}
    assert rows["depth35-live-seam-abce"]["audit"]["sha256_forward"] == (
        "78f8d4f17f2f204fee7619caefa70e28442ff6a0e09e3da263b01d0659bc17be"
    )
    assert rows["depth39-live-seam-abce"]["audit"]["sha256_forward"] == (
        "789121ad512b409e9d655cd8e62a56fb18ca8647061cc1027dbec4b401fabb80"
    )
