import time

from experiments import argonne_connected_events_20261009 as connected_events
from experiments.argonne_connected_events_20261009 import (
    Event, Operation, World, build_events, candidate_record, closest_boundary_events, phrase_reuse_audit, replay, search,
)


def test_signing_requires_current_holder_to_read():
    world = World(2, 1)
    read_a, sign_a = Operation("read", 0, 0), Operation("sign", 0, 0)
    assert world.apply(world.initial, sign_a) is None
    read_state = world.apply(world.initial, read_a)
    assert world.goal(world.apply(read_state, sign_a))
    transferred = world.apply(read_state, Operation("handover", 0, 0, 1))
    assert world.apply(transferred, sign_a) is None
    assert world.apply(transferred, Operation("sign", 0, 1)) is None
    reread = world.apply(transferred, Operation("read", 0, 1))
    assert world.goal(world.apply(reread, Operation("sign", 0, 1)))


def test_backward_preimage_matches_exhaustive_forward_transition():
    world = World(2, 1)
    for operation in (Operation("read", 0, 0), Operation("sign", 0, 1), Operation("handover", 0, 0, 1)):
        for allowed in (world.goal_states, 1, 3, (1 << world.states) - 1):
            preimage = world.preimage(operation, allowed)
            for before in range(world.states):
                after = world.apply(before, operation)
                assert bool(preimage & (1 << before)) == (after is not None and bool(allowed & (1 << after)))


def test_different_wordings_cannot_repeat_the_same_proposition():
    world = World(1, 1)
    read = Operation("read", 0, 0)
    rows = [Event("Ada reads the form", read, "active"), Event("The form is read by Ada", read, "passive")]
    assert replay(rows, world)["reason"] == "repeated_proposition"


def test_grounded_trace_connects_read_to_signature():
    world = World(2, 1)
    rows = [Event("Ada hands the form to Eve", Operation("handover", 0, 0, 1), "handover"),
            Event("Eve reads the form", Operation("read", 0, 1), "read"),
            Event("Eve signs the form", Operation("sign", 0, 1), "sign")]
    result = replay(rows, world)
    assert result["valid"] and result["goal_reached"] and result["connected_events"]
    assert {"from": 1, "to": 2, "kind": "read_before_sign"} in result["causal_links"]
    assert not candidate_record(rows, world)["exact"]


def test_exact_search_closes_tiny_synthetic_language_with_real_preconditions():
    # Artificial one-letter surfaces exercise mechanics, never prose evidence.
    world = World(1, 1)
    rows = [Event("a", Operation("read", 0, 0), "fixture_read"),
            Event("a", Operation("sign", 0, 0), "fixture_sign")]
    result = search(rows, world, min_letters=2, max_letters=2,
                    deadline=time.monotonic() + 0.1, max_states=50)
    assert result["candidates"]
    candidate = result["candidates"][0]
    assert candidate["exact"] and candidate["letters"] == 2
    assert candidate["semantics"]["goal_reached"]
    assert candidate["construction_flags"]["repeated_sentences"]
    assert not search(rows, world, min_letters=3, max_letters=3,
                      deadline=time.monotonic() + 0.1, max_states=50)["candidates"]


def test_center_can_fall_inside_an_event_without_aligned_clause_boundaries():
    world = World(1, 1)
    rows = [Event("ab", Operation("read", 0, 0), "fixture_read"),
            Event("a", Operation("sign", 0, 0), "fixture_sign")]
    result = search(rows, world, min_letters=3, max_letters=3,
                    deadline=time.monotonic() + 0.1, max_states=50)
    assert result["candidates"][0]["exact"]
    assert result["candidates"][0]["text"] == "Ab. A."
    assert result["candidates"][0]["semantics"]["goal_reached"]


def test_grammar_keeps_aliases_and_roles_and_excludes_old_observation_chains():
    rows = build_events(("Ada", "Eve"), ("form",))
    assert len({row.operation for row in rows}) == 6
    assert any(row.template == "passive_sign" for row in rows)
    assert not any(any(word in row.text.lower().split() for word in ("sees", "spots", "stops")) for row in rows)


def test_zero_result_explains_boundary_conflict_and_event_support():
    world = World(1, 1)
    rows = [Event("ab", Operation("read", 0, 0), "read_fixture"),
            Event("c", Operation("sign", 0, 0), "sign_fixture")]
    result = search(rows, world, min_letters=3, max_letters=3,
                    deadline=time.monotonic() + 0.1, max_states=50)
    assert result["candidates"] == []
    assert result["termination"] == "frontier_exhausted"
    assert result["event_coverage"]["max_events_in_partial_state"] == 1
    assert result["event_coverage"]["max_forward_read_documents"] == 1
    frontier = result["frontier"][0]
    assert frontier["unmatched_letters"] == "ab"
    assert frontier["required_side"] == "right"
    assert frontier["prune_reasons"]["no_letter_compatible_event"]
    assert frontier["closest_boundary_events"][0]["first_mismatch"]["offset"] == 0
    assert result["counters"]["states_popped"] >= result["states"]


def test_boundary_diagnostic_uses_right_suffix_in_reverse_direction():
    events = [Event("cab", Operation("read", 0, 0), "fixture")]
    diagnostic, = closest_boundary_events(events, "bad", "right")
    assert diagnostic["common_prefix_letters"] == 2
    assert diagnostic["first_mismatch"] == {"offset": 2, "required": "d", "proposed": "c"}
    assert diagnostic["semantic_eligibility"] == "not_checked"


def test_state_and_time_limits_are_not_reported_as_exhaustive_failure():
    world = World(1, 1)
    rows = [Event("a", Operation("read", 0, 0), "read"), Event("a", Operation("sign", 0, 0), "sign")]
    capped = search(rows, world, min_letters=2, max_letters=2,
                    deadline=time.monotonic() + 0.1, max_states=1)
    assert capped["termination"] == "state_limit"
    assert capped["remaining_queue"] > 0
    expired = search(rows, world, min_letters=2, max_letters=2,
                     deadline=time.monotonic() - 1, max_states=50)
    assert expired["termination"] == "deadline"
    assert expired["states"] == 0


def test_deadline_during_expansion_does_not_claim_exhausted_boundary(monkeypatch):
    ticks = iter((0.0, 1.0))
    monkeypatch.setattr(connected_events.time, "monotonic", lambda: next(ticks, 1.0))
    rows = [Event("a", Operation("read", 0, 0), "read")]
    result = search(rows, World(1, 1), min_letters=1, max_letters=3, deadline=0.5)
    assert result["termination"] == "deadline"
    assert not result["frontier"][0]["expansion_complete"]
    assert result["frontier"][0]["prune_reasons"] == {"deadline_incomplete_expansion": 1}


def test_rewording_same_operation_is_not_new_progress():
    operation = Operation("read", 0, 0)
    rows = [Event("Ada reads the form", operation, "active_read"),
            Event("The form is read by Ada", operation, "passive_read")]
    audit = phrase_reuse_audit(rows, World(1, 1))
    assert not audit["passed"]
    assert audit["repeated_propositions"][0]["count"] == 2
    assert audit["grounded_operation_count"] == 1


def test_same_actor_frame_and_object_reference_can_ground_distinct_events():
    rows = [Event("Ada reads the form", Operation("read", 0, 0), "active_read"),
            Event("Ada signs the form", Operation("sign", 0, 0), "active_sign"),
            Event("Ada reads the letter", Operation("read", 1, 0), "active_read"),
            Event("Ada signs the letter", Operation("sign", 1, 0), "active_sign")]
    audit = phrase_reuse_audit(rows, World(1, 2))
    assert audit["passed"] and audit["grounded_operation_count"] == 4
    repeated = {row["phrase"]: row for row in audit["repeated_predicate_phrases"]}
    assert repeated["ada reads the"]["licensed"]
    assert repeated["ada signs the"]["distinct_operation_bindings"]
    assert any(row["phrase"] == "the form" for row in audit["allowed_reference_phrases"])
    assert audit["event_groundings"][1]["effect"]["after"]["signed"]


def test_same_predicate_phrase_can_bind_different_current_holders():
    rows = [Event("Ada reads the form", Operation("read", 0, 0), "active_read"),
            Event("Ada hands the form to Eve", Operation("handover", 0, 0, 1), "hands_to"),
            Event("Eve reads the form", Operation("read", 0, 1), "active_read"),
            Event("Eve signs the form", Operation("sign", 0, 1), "active_sign")]
    world = World(2, 1)
    audit = phrase_reuse_audit(rows, world)
    assert audit["passed"] and replay(rows, world)["goal_reached"]
    repeated = {row["phrase"]: row for row in audit["repeated_predicate_phrases"]}
    assert repeated["reads the form"]["licensed"]
    assert audit["event_groundings"][0]["role_bindings"]["agent"] != audit["event_groundings"][2]["role_bindings"]["agent"]
