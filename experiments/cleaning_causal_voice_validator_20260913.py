"""Independent cleaning-state grammar with structural pronoun resolution."""
from __future__ import annotations

HUMAN_PL = set("workers owners cooks chefs servers cleaners helpers attendants".split())
ARTIFACT_PL = set("ovens trays counters plates pans grills stoves tables boards bowls aprons towels".split())
ARTIFACT_SG = set("oven tray counter plate pan grill stove table board bowl apron towel".split())
CLASSIFIERS = {"deli", "cafe", "kitchen"}
HUMAN_ADJ = {"tired", "careful", "skilled"}
ARTIFACT_ADJ = {"small", "large", "old"}
ACTIVE_EFFECTS = {"wash": -1, "clean": -1, "scrub": -1, "wipe": -1, "soil": 1}
PASSIVE_ACTION = {"washed": "wash", "cleaned": "clean", "scrubbed": "scrub", "wiped": "wipe", "soiled": "soil"}
DIRT_STATE = {"dirty": 1, "soiled": 2, "stained": 2, "dusty": 1, "clean": 0}


def resolve_pronoun(token, case, entities, required_type="artifact"):
    forms = {("it", "subject"): "singular", ("it", "object"): "singular",
             ("they", "subject"): "plural", ("them", "object"): "plural"}
    number = forms.get((token, case))
    compatible = [entity for entity in entities if entity["type"] == required_type and entity["number"] == number]
    if len(compatible) != 1:
        return None
    return {**compatible[0], "resolved_form": token, "resolved_case": case,
            "resolution": "unique antecedent matching grammatical case, number and artifact type"}


def noun_phrases(words, start, kind, identity):
    out = []
    heads = HUMAN_PL if kind == "human" else ARTIFACT_PL | ARTIFACT_SG
    modifiers = HUMAN_ADJ if kind == "human" else ARTIFACT_ADJ
    for end in range(start + 1, min(len(words), start + 3) + 1):
        chunk = words[start:end]
        if not chunk or chunk[-1] not in heads:
            continue
        head, lead = chunk[-1], list(chunk[:-1])
        number = "singular" if head in ARTIFACT_SG else "plural"
        det = lead.pop(0) if lead and lead[0] in {"a", "an", "the", "some"} else None
        if len(lead) > 1 or (lead and lead[0] not in modifiers | CLASSIFIERS):
            continue
        if number == "singular" and det not in {"a", "an", "the"}:
            continue
        if number == "plural" and det in {"a", "an"}:
            continue
        if det in {"a", "an"}:
            initial = lead[0] if lead else head
            if det != ("an" if initial[0] in "aeiou" else "a"):
                continue
        out.append((end, {"id": identity, "type": kind, "number": number, "head": head,
                          "surface": chunk, "syntactic_role": "agent" if kind == "human" else "patient"}))
    return out


def human_relative_end(words, start):
    if start >= len(words) or words[start] != "who":
        return start, None
    i = start + 1
    if i >= len(words) or words[i] not in {"read", "review", "follow"}:
        return None, None
    verb = words[i]
    i += 1
    if i < len(words) and words[i] in {"brief", "detailed", "clear"}:
        i += 1
    if i >= len(words) or words[i] not in {"instructions", "manuals", "guidelines"}:
        return None, None
    obj = words[i]
    i += 1
    if i < len(words) and words[i] == "from":
        i += 1
        if i < len(words) and words[i] in {"careful", "skilled"}:
            i += 1
        if i >= len(words) or words[i] not in {"supervisors", "managers", "inspectors"}:
            return None, None
        i += 1
    return i, {"verb": verb, "object": obj, "subject_shared_with_agent": True}


def finish_location(words, start):
    if start == len(words):
        return True, None
    rest = words[start:]
    if len(rest) == 3 and rest[0] in {"in", "inside", "near"} and rest[1] in {"small", "large", "quiet"} and rest[2] in {"kitchens", "cafes", "workshops"}:
        return True, {"preposition": rest[0], "place": rest[2]}
    return False, None


def condition(words, start, entities, introduce):
    if introduce:
        targets = noun_phrases(words, start, "artifact", "condition-patient")
    else:
        target = resolve_pronoun(words[start], "subject", entities) if start < len(words) else None
        targets = [(start + 1, target)] if target else []
    out = []
    for end, target in targets:
        copula = "is" if target["number"] == "singular" else "are"
        if end + 1 < len(words) and words[end] == copula and words[end + 1] in DIRT_STATE:
            out.append((end + 2, {"patient": target, "property": "surface_contamination", "state_word": words[end + 1],
                                  "state": DIRT_STATE[words[end + 1]], "copula": copula}))
    return out


def actions(words, entities):
    out = []
    for end_agent, agent in noun_phrases(words, 0, "human", "action-agent"):
        after_relative, relative = human_relative_end(words, end_agent)
        if after_relative is None or after_relative >= len(words) or words[after_relative] not in ACTIVE_EFFECTS:
            continue
        verb = words[after_relative]
        start_target = after_relative + 1
        targets = noun_phrases(words, start_target, "artifact", "action-patient")
        if start_target < len(words):
            target = resolve_pronoun(words[start_target], "object", entities + [agent])
            if target:
                targets.append((start_target + 1, target))
        for end_target, target in targets:
            complete, location = finish_location(words, end_target)
            if complete:
                out.append({"voice": "active", "agent": agent, "patient": target, "verb": verb,
                            "relative": relative, "location": location, "entities": entities + [agent, target]})
    targets = noun_phrases(words, 0, "artifact", "action-patient")
    if words:
        target = resolve_pronoun(words[0], "subject", entities)
        if target:
            targets.append((1, target))
    for end_target, target in targets:
        copula = "is" if target["number"] == "singular" else "are"
        if end_target + 2 >= len(words) or words[end_target] != copula or words[end_target + 1] not in PASSIVE_ACTION or words[end_target + 2] != "by":
            continue
        verb = PASSIVE_ACTION[words[end_target + 1]]
        for end_agent, agent in noun_phrases(words, end_target + 3, "human", "action-agent"):
            after_relative, relative = human_relative_end(words, end_agent)
            if after_relative is None:
                continue
            complete, location = finish_location(words, after_relative)
            if complete:
                out.append({"voice": "passive", "agent": agent, "patient": target, "verb": verb,
                            "relative": relative, "location": location, "entities": entities + [target, agent]})
    return out


def causal_witness(action, observation):
    same_patient = action["patient"]["id"] == observation["patient"]["id"]
    initial = observation["state"]
    effect = ACTIVE_EFFECTS[action["verb"]]
    predicted = max(0, initial + effect)
    return {"valid": same_patient and predicted < initial,
            "same_patient_identity": same_patient, "patient_id": action["patient"]["id"],
            "observable_property": "surface_contamination", "initial_contamination": initial,
            "action_effect": effect, "predicted_contamination": predicted,
            "counterfactual_test": "the action lowers the stated contamination of that same artifact"}


def parse_complete(words):
    words = tuple(words)
    if words.count("because") != 1:
        return []
    pairs = []
    if words[0] == "because":
        for end, observation in condition(words, 1, [], True):
            for action in actions(words[end:], [observation["patient"]]):
                pairs.append(("condition_first", action, observation))
    else:
        split = words.index("because")
        for action in actions(words[:split], []):
            for end, observation in condition(words, split + 1, action["entities"], False):
                if end == len(words):
                    pairs.append(("reason_last", action, observation))
    return [{"order": order, "voice": action["voice"], "action": action, "observation": observation,
             "causal_witness": causal_witness(action, observation), "complete": True,
             "semantic_relation_valid": causal_witness(action, observation)["valid"]} for order, action, observation in pairs]
