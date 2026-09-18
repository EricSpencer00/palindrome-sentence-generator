"""Independent complete-sentence grammar for an art-conservation event.

Nothing is imported from the generating grammar or lattice.  The qualitative
damage model tests whether the chosen action changes the stated defect of
the same patient; it does not certify necessity, elegance, or reader quality.
"""
from __future__ import annotations

PEOPLE = frozenset("artists painters sculptors framers restorers conservators curators dealers traders collectors printmakers woodcarvers".split())
COMPOUND_PEOPLE = {("print", "makers"), ("wood", "carvers")}
PATIENTS = {
    "art": "mixed", "artwork": "mixed", "paintings": "painted",
    "prints": "paper", "drawings": "paper", "posters": "paper",
    "woodcuts": "wood", "sculptures": "solid", "frames": "solid",
}
SPACED_PATIENTS = {("art", "work"): "mixed", ("wood", "cuts"): "wood"}
DEFECTS = {"torn": ("tear", 2), "chipped": ("chip", 2),
           "faded": ("fading", 1), "damaged": ("damage", 2)}
SURFACE_PROPERTIES = {"tear": {"paper", "painted", "mixed"},
                      "chip": {"solid", "wood", "painted", "mixed"},
                      "fading": {"paper", "wood", "painted", "mixed"},
                      "damage": set(PATIENTS.values())}
EFFECTS = {"restore": {"tear": -1, "chip": -1, "fading": -1, "damage": -1},
           "repair": {"tear": -1, "chip": -1, "damage": -1},
           "mend": {"tear": -1}, "damage": {"damage": 1}}


def parse_sentence(words):
    """Reparse one whole surface, with no prefix/suffix or source analyses."""
    words = tuple(words)
    actors = []
    if words and words[0] in PEOPLE:
        actors.append((1, words[:1]))
    if words[:2] in COMPOUND_PEOPLE:
        actors.append((2, words[:2]))
    analyses = []
    for position, actor in actors:
        relative = None
        if position < len(words) and words[position] == "who":
            start = position
            position += 1
            if position >= len(words) or words[position] not in {"study", "consult", "compare"}:
                continue
            position += 1
            if position < len(words) and words[position] in {"old", "detailed", "archival"}:
                position += 1
            if position >= len(words) or words[position] not in {"records", "photographs", "catalogues"}:
                continue
            position += 1
            if position < len(words) and words[position] == "from":
                position += 1
                if position < len(words) and words[position] in {"local", "national", "private"}:
                    position += 1
                if position >= len(words) or words[position] not in {"museums", "archives", "collections"}:
                    continue
                position += 1
            relative = {"tokens": words[start:position], "agent_id": "actor"}
        manner = None
        if position < len(words) and words[position] in {"carefully", "patiently", "slowly"}:
            manner = words[position]
            position += 1
        if position >= len(words) or words[position] not in EFFECTS:
            continue
        verb = words[position]
        position += 1
        if position >= len(words) or words[position] not in DEFECTS:
            continue
        defect_word = words[position]
        property_name, initial = DEFECTS[defect_word]
        position += 1
        color = None
        if position < len(words) and words[position] in {"red", "blue", "green", "black"}:
            color = words[position]
            position += 1
        objects = []
        if position < len(words) and words[position] in PATIENTS:
            objects.append((position + 1, words[position:position + 1], PATIENTS[words[position]]))
        if words[position:position + 2] in SPACED_PATIENTS:
            objects.append((position + 2, words[position:position + 2], SPACED_PATIENTS[words[position:position + 2]]))
        for end, patient, medium in objects:
            location = None
            if end != len(words):
                remainder = words[end:]
                if len(remainder) != 3 or remainder[0] not in {"in", "inside", "near"} or remainder[1] not in {"local", "quiet", "private"} or remainder[2] not in {"museums", "galleries", "studios"}:
                    continue
                location = remainder
            effect = EFFECTS[verb].get(property_name)
            predicted = max(0, initial + effect) if effect is not None else initial
            compatible = medium in SURFACE_PROPERTIES[property_name]
            witness = {"patient_id": "patient", "observed_patient_id": "patient",
                       "property": property_name, "initial": initial, "action_effect": effect,
                       "predicted": predicted, "medium_supports_observation": compatible,
                       "valid": compatible and effect is not None and predicted < initial}
            analyses.append({"complete": True, "actor": actor, "actor_id": "actor", "verb": verb,
                             "patient": patient, "patient_id": "patient", "medium": medium,
                             "defect": defect_word, "color": color, "relative": relative,
                             "manner": manner, "location": location,
                             "event_effect_witness": witness, "semantic_relation_valid": witness["valid"]})
    return analyses
