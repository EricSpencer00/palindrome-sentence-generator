"""Separately declared grammar and qualitative causal-state audit.

This parser imports no generator declarations. Word membership recognizes
syntax; causal validation separately resolves the contents/container relation
and computes whether the proposed action reduces the stated temperature
deviation. The numeric values are qualitative signs, not physical units or a
claim that an action is necessary in every real-world setting.
"""
from __future__ import annotations

ACTORS = set("cooks chefs servers parents doctors nurses workers helpers caretakers attendants volunteers hosts guests".split())
ACTOR_ADJ = {"careful", "skilled", "tired"}
VESSELS = set("bowls cups pots pans plates trays jars".split())
VESSEL_ADJ = {"small", "large", "deep"}
FOOD = set("soup stew sauce curry rice pasta cod trout salmon chicken beef pork beans peas carrots potatoes noodles grits porridge pudding".split())
HEAT_STATE = {"hot": 1, "cold": -1}
ACTION_DELTA = {"cool": -1, "chill": -1, "warm": 1, "heat": 1}
MANNER = {"carefully", "slowly", "gently"}


def causal_state_witness(action, temperature, target_type="container", condition_owner="contents", referent_bound=True):
    """Infer the reason relation from state, containment and action effects."""
    if action not in ACTION_DELTA or temperature not in HEAT_STATE:
        return {"valid": False, "reason": "unknown_state_or_action"}
    if target_type != "container" or condition_owner != "contents" or not referent_bound:
        return {"valid": False, "reason": "condition_does_not_describe_target_contents"}
    contents_state = HEAT_STATE[temperature]
    # The stated contents are the relevant thermal source for this container.
    # The action's sign must correct, not amplify, that stated deviation.
    inferred_container_state = contents_state
    action_effect = ACTION_DELTA[action]
    predicted = inferred_container_state + action_effect
    improves = abs(predicted) < abs(inferred_container_state)
    return {"valid": improves, "model": "signed_temperature_correction_through_contents",
            "condition_contents_state": contents_state, "inferred_container_state": inferred_container_state,
            "action_effect": action_effect, "predicted_state": predicted, "target_state": 0,
            "initial_deviation": abs(inferred_container_state), "predicted_deviation": abs(predicted),
            "referent_bound": referent_bound, "causal_test": "predicted_deviation < initial_deviation",
            "reason": "action_reduces_stated_thermal_deviation" if improves else "action_worsens_stated_thermal_deviation"}


def parse_complete(words):
    words = tuple(words)
    if words.count("because") != 1:
        return []
    boundary = words.index("because")
    left, reason = words[:boundary], words[boundary + 1:]
    if len(reason) not in (5, 6) or reason[:3] != ("their", "contents", "include"):
        return []
    reason_np = reason[3:]
    if len(reason_np) == 3:
        if reason_np[0] != "very":
            return []
        reason_np = reason_np[1:]
    temperature, food = reason_np
    if temperature not in HEAT_STATE or food not in FOOD:
        return []
    i = 0
    if left and left[0] in ACTOR_ADJ:
        i += 1
    if i >= len(left) or left[i] not in ACTORS:
        return []
    actor = left[i]
    i += 1
    relative = None
    if i < len(left) and left[i] == "who":
        start = i
        i += 1
        if i < len(left) and left[i] in MANNER:
            i += 1
        if i >= len(left) or left[i] not in {"read", "reviewed", "checked"}:
            return []
        relative_verb = left[i]
        i += 1
        if i < len(left) and left[i] in {"brief", "detailed", "old"}:
            i += 1
        if i >= len(left) or left[i] not in {"recipes", "menus", "instructions"}:
            return []
        relative_object = left[i]
        i += 1
        if i < len(left) and left[i] == "from":
            i += 1
            if i < len(left) and left[i] in {"skilled", "experienced", "careful"}:
                i += 1
            if i >= len(left) or left[i] not in {"cooks", "chefs", "hosts"}:
                return []
            i += 1
        relative = {"subject": actor, "verb": relative_verb, "object": relative_object,
                    "shared_subject": True, "complete": True, "token_span": [start, i]}
    if i < len(left) and left[i] in MANNER:
        i += 1
    if i >= len(left) or left[i] not in ACTION_DELTA:
        return []
    action = left[i]
    i += 1
    if i < len(left) and left[i] in {"the", "some"}:
        i += 1
    if i < len(left) and left[i] in VESSEL_ADJ:
        i += 1
    if i >= len(left) or left[i] not in VESSELS:
        return []
    vessel = left[i]
    i += 1
    location = None
    if i < len(left) and left[i] in {"in", "inside", "near"}:
        i += 1
        if i < len(left) and left[i] in {"quiet", "large", "old"}:
            i += 1
        if i >= len(left) or left[i] not in {"kitchens", "cafeterias", "diningrooms"}:
            return []
        location = left[i]
        i += 1
    if i != len(left):
        return []
    causal = causal_state_witness(action, temperature)
    return [{"complete": True, "subject": actor, "subject_number": "plural", "verb_form": "present_base",
             "action": action, "target": {"word": vessel, "type": "container", "number": "plural"},
             "relative": relative, "event_location": location,
             "reason": {"observable_property": "contents_temperature", "temperature": temperature, "food": food,
                        "possessor": vessel, "contents_of_action_target": True},
             "causal_witness": causal, "agreement_ok": True, "valency_ok": True,
             "semantic_relation_valid": causal["valid"]}]
