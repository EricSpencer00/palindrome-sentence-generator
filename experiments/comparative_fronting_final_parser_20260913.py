"""Separately authored final comparative-inversion grammar and scalar reparse.

The explicit ``than before`` compares two states of the postposed subject.
The repair process is controlled by that same patient, not by the apparent
position of the fronted adjective or the source constructor's annotations.
"""
from __future__ import annotations


def parse_final(tokens):
    words = tuple(tokens)
    concessive = words[:3] == ("none", "the", "less")
    first = 3 if concessive else 0
    measures = {("smoother",): ("surface_roughness", -1), ("less", "permeable"): ("permeability", -1),
                ("firmer",): ("joint_stability", 1), ("more", "secure"): ("joint_stability", 1),
                ("lower",): ("ridge_height", -1)}
    treatments = {"polishing": {"dimension": "surface_roughness", "delta": -2, "means": ("fine", "abrasives")},
                  "sealing": {"dimension": "permeability", "delta": -2, "means": ("fresh", "adhesive")},
                  "tightening": {"dimension": "joint_stability", "delta": 2, "means": ("steel", "screws")},
                  "sanding": {"dimension": "ridge_height", "delta": -2, "means": ("coarse", "abrasives")},
                  "loosening": {"dimension": "joint_stability", "delta": -1, "means": ("steel", "screws")}}
    conditions = {("rough",): ("surface_roughness", 3, False), ("leaky",): ("permeability", 2, False),
                  ("loose", "joints", "in"): ("joint_stability", 1, True),
                  ("reworked", "ridges", "on"): ("ridge_height", 3, True)}
    patients = {("metal", "work"): ("metal", False), ("wood", "work"): ("wood", False),
                ("glass", "work"): ("glass", False), ("oak", "panels"): ("wood", True),
                ("wooden", "vessel"): ("wood", False)}
    try:
        marker = words.index("than", first)
    except ValueError:
        return []
    comparative = words[first:marker]
    if comparative not in measures or words[marker:marker + 3] != ("than", "before", "after"):
        return []
    repair_position = marker + 3
    if repair_position >= len(words) or words[repair_position] not in treatments:
        return []
    action = words[repair_position]
    treatment = treatments[action]
    required = ("with",) + treatment["means"] + ("under", "steady", "pressure", "for", "several", "hours")
    remainder = words[repair_position + 1:]
    if remainder[:len(required)] != required:
        return []
    post = remainder[len(required):]
    if len(post) < 3 or post[0] not in {"is", "are"} or post[1] != "the":
        return []
    subject = post[2:]
    analyses = []
    for condition, (dimension, before, plural_feature) in conditions.items():
        for material_tokens, (material, plural_material) in patients.items():
            if subject != condition + material_tokens or post[0] != ("are" if plural_feature or plural_material else "is"):
                continue
            measured_dimension, sign = measures[comparative]
            predicted = max(0, before + treatment["delta"]) if treatment["dimension"] == dimension else before
            comparison_true = predicted < before if sign < 0 else predicted > before
            dimension_agrees = dimension == treatment["dimension"] == measured_dimension
            material_valid = dimension != "joint_stability" or material in {"metal", "wood"}
            analyses.append({"complete": True, "concessive": concessive,
                "patient": {"id": "postposed_subject", "tokens": subject, "material": material,
                            "number": "plural" if plural_feature or plural_material else "singular"},
                "initial_property": {"dimension": dimension, "value": before, "patient_id": "postposed_subject"},
                "repair_process": {"action": action, "dimension": treatment["dimension"], "effect": treatment["delta"],
                                   "controlled_patient_id": "postposed_subject"},
                "predicted_state": predicted,
                "fronted_comparative": {"tokens": comparative, "dimension": measured_dimension,
                                        "direction": "decrease" if sign < 0 else "increase",
                                        "current_patient_id": "postposed_subject", "prior_patient_id": "postposed_subject",
                                        "prior_state": before, "current_state": predicted, "satisfied": comparison_true},
                "causal_witness": {"same_patient": True, "same_property": dimension_agrees,
                                   "material_compatible": material_valid, "ordered_change": comparison_true},
                "render_boundaries": {"after_before": marker + 1, "after_process": repair_position + len(required)},
                "semantic_relation_valid": dimension_agrees and material_valid and comparison_true})
    return analyses


def render_final(words):
    rendered = list(words)
    parsed = parse_final(words)
    if parsed:
        if parsed[0]["concessive"]:
            rendered[2] += ","
        for index in parsed[0]["render_boundaries"].values():
            rendered[index] += ","
    return " ".join(rendered).capitalize() + "."
