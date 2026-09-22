from experiments.semantic_event_lattice_multiedge_20260920 import audit, edge_ok, run


def test_three_edge_states_are_typed_and_intact():
    result = run()
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["event_edges"] == 3
    assert result["rendered_controls"]
    assert all(row["complete_prose"] and len(row["edges"]) == 3 for row in result["rendered_controls"])
    assert all(row["provenance"]["no_shortcuts"] if "no_shortcuts" in row["provenance"] else True for row in result["rendered_controls"])


def test_live_edge_types_reject_bad_focus_or_actor_chain():
    rows = run()["rendered_controls"]
    assert all(row["edges"][0]["focus"] == "arrival" for row in rows)
    assert not edge_ok(rows[0]["edges"][0], {"actor": "Mara", "verb": "offered", "recipient": "Ivo", "theme": "a map", "focus": "handoff"})


def test_exactness_hash_is_independently_reported_and_reader_gate_closed():
    result = run()
    assert result["reader_eligible"] is False
    for row in result["rendered_controls"]:
        checked = audit(row["rendered"])
        assert checked["sha256_forward"] == row["audit"]["sha256_forward"]
        assert checked["sha256_reverse"] == row["audit"]["sha256_reverse"]
    assert result["provenance"]["no_shortcuts"]
