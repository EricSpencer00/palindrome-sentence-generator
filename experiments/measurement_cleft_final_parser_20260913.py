"""Separately authored final measurement-cleft grammar and scalar interpreter.

The relative measure, equated feature/measure, material patient and final
finite repair predicate are structurally unified.  This model is a bounded
causal plausibility check, not evidence of reader quality or originality.
"""
from __future__ import annotations


def parse_final(tokens):
    words = tuple(tokens)
    concessive = words[:3] == ("none", "the", "less")
    offset = 3 if concessive else 0
    dimensions = {
        "height": {"initial": 3, "direction": "lower", "generic": {"sections", "sectors"}, "entity": ("ridges",),
                   "measurement": ("ridge", "height"), "cause": ("coarse", "sand", "paper"),
                   "entity_verbs": {"reduces": -2, "cuts": -2, "raises": 2}, "measurement_verbs": {"reduces": -2, "raises": 2}},
        "roughness": {"initial": 3, "direction": "lower", "generic": {"surfaces", "patches"}, "entity": ("areas",),
                      "measurement": ("surface", "roughness"), "cause": ("fine", "sand", "paper"),
                      "entity_verbs": {"smooths": -2, "roughens": 2}, "measurement_verbs": {"reduces": -2, "raises": 2}},
        "firmness": {"initial": 1, "direction": "higher", "generic": {"joints", "seams"}, "entity": ("connections",),
                     "measurement": ("joint", "firmness"), "cause": ("fresh", "adhesive"),
                     "entity_verbs": {"secures": 2, "weakens": -1}, "measurement_verbs": {"increases": 2, "reduces": -2}},
        "permeability": {"initial": 2, "direction": "lower", "generic": {"slabs", "panels"}, "entity": ("plates",),
                         "measurement": ("surface", "permeability"), "cause": ("fresh", "adhesive"),
                         "entity_verbs": {"seals": -2, "opens": 2}, "measurement_verbs": {"reduces": -2, "raises": 2}},
    }
    material_terms = {("red", "metal", "work"): "metal", ("old", "wood", "work"): "wood", ("clear", "glass", "work"): "glass"}
    rows = []
    for property_name, specification in dimensions.items():
        for mode in ("scalar_what_cleft", "headed_relative_equative"):
            if mode == "scalar_what_cleft":
                beginning = ("what", "is", specification["direction"])
                identity = specification["measurement"]
                copula, verbs = "is", specification["measurement_verbs"]
                first_head = "what"
            else:
                if offset >= len(words) or words[offset] not in specification["generic"]:
                    continue
                first_head = words[offset]
                beginning = (first_head, "whose", property_name, "is", specification["direction"])
                identity = specification["entity"]
                copula, verbs = "are", specification["entity_verbs"]
            if words[offset:offset + len(beginning)] != beginning:
                continue
            index = offset + len(beginning)
            comparison = ("than", "before", "the", "recent", "repair", "on", "the")
            if words[index:index + len(comparison)] != comparison:
                continue
            index += len(comparison)
            for material_np, material in material_terms.items():
                equated = material_np + (copula, "the") + identity + ("that",) + specification["cause"]
                if words[index:index + len(equated)] != equated:
                    continue
                ending = words[index + len(equated):]
                if len(ending) != 1 or ending[0] not in verbs:
                    continue
                action = ending[0]
                before, effect = specification["initial"], verbs[action]
                predicted = max(0, before + effect)
                ordered = predicted < before if specification["direction"] == "lower" else predicted > before
                rows.append({"complete": True, "mode": mode, "concessive": concessive,
                    "patient": {"id": "feature_on_material", "feature_head": specification["entity"], "material": material},
                    "initial_relative": {"head": first_head, "property": property_name, "before": before,
                                         "predicted": predicted, "patient_id": "feature_on_material"},
                    "equative_binding": {"left_patient_id": "feature_on_material", "right_patient_id": "feature_on_material",
                                          "right_kind": "scalar_measure" if mode == "scalar_what_cleft" else "physical_feature", "right_tokens": identity},
                    "finite_repair_relative": {"subject": specification["cause"], "predicate": action,
                                               "gap_patient_id": "feature_on_material", "affected_property": property_name, "effect": effect},
                    "measured_comparison": {"property": property_name, "before_patient_id": "feature_on_material",
                                            "after_patient_id": "feature_on_material", "before": before, "after": predicted,
                                            "direction": specification["direction"], "satisfied": ordered},
                    "semantic_relation_valid": ordered})
    return rows


def render_final(words):
    result = list(words)
    parsed = parse_final(words)
    if parsed and parsed[0]["concessive"]:
        result[2] += ","
    return " ".join(result).capitalize() + "."
