"""Separately authored whole-sentence scalar-result grammar.

The participial repair and final predicate are controlled by the same subject.
A numeric qualitative state model checks the effect direction and measured
result; accepting the syntax alone never certifies the causal relation.
"""
from __future__ import annotations


def parse_final(tokens):
    words = tuple(tokens)
    patients = {("metal", "work"): {"substrate": "metal", "plural": False},
                ("wood", "work"): {"substrate": "wood", "plural": False},
                ("glass", "work"): {"substrate": "glass", "plural": False},
                ("oak", "panels"): {"substrate": "wood", "plural": True}}
    initial_phrases = {("rough",): {"dimension": "surface_roughness", "value": 3, "plural": False},
                       ("leaky",): {"dimension": "gas_flow", "value": 2, "plural": False},
                       ("loose", "joints", "in"): {"dimension": "joint_stability", "value": 1, "plural": True},
                       ("reworked", "ridges", "on"): {"dimension": "ridge_height", "value": 3, "plural": True}}
    operations = {"polished": ("surface_roughness", -2, ("fine", "abrasives")),
                  "sealed": ("gas_flow", -2, ("fresh", "adhesive")),
                  "tightened": ("joint_stability", 2, ("steel", "screws")),
                  "sanded": ("ridge_height", -2, ("coarse", "abrasives")),
                  "loosened": ("joint_stability", -1, ("steel", "screws")),
                  "roughened": ("surface_roughness", 2, ("fine", "abrasives"))}
    predicates = {("smoother",): ("surface_roughness", "less"),
                  ("smoother", "than", "before"): ("surface_roughness", "less"),
                  ("air", "tight"): ("gas_flow", "none"),
                  ("less", "permeable"): ("gas_flow", "less"),
                  ("firmer",): ("joint_stability", "more"),
                  ("more", "secure"): ("joint_stability", "more"),
                  ("lower",): ("ridge_height", "less"),
                  ("lower", "than", "before"): ("ridge_height", "less")}
    analyses = []
    for prefix, initial in initial_phrases.items():
        for material_np, matter in patients.items():
            subject = prefix + material_np
            if words[:len(subject)] != subject:
                continue
            position = len(subject)
            source_clause = ("that", "arrived", "from", "regional", "museums")
            if words[position:position + 5] != source_clause:
                continue
            position += 5
            participle_start = position
            if position >= len(words) or words[position] not in operations:
                continue
            repair = words[position]
            dimension, effect, applicator = operations[repair]
            tail = ("with",) + applicator + ("under", "steady", "pressure")
            position += 1
            if words[position:position + len(tail)] != tail:
                continue
            position += len(tail)
            participle_end = position
            is_plural = initial["plural"] or matter["plural"]
            if words[position:position + 2] != (("are" if is_plural else "is"), "now"):
                continue
            result = words[position + 2:]
            if result not in predicates:
                continue
            stated_dimension, relation = predicates[result]
            prior = initial["value"]
            predicted = max(0, prior + effect) if dimension == initial["dimension"] else prior
            relation_true = (predicted < prior if relation == "less" else predicted > prior if relation == "more" else predicted == 0)
            material_support = initial["dimension"] != "joint_stability" or matter["substrate"] in {"metal", "wood"}
            same_dimension = dimension == initial["dimension"] == stated_dimension
            analyses.append({"complete": True, "patient": {"id": "subject_patient", "subject_tokens": subject,
                             "host_material": matter["substrate"], "plural": is_plural},
                             "initial_property": {"dimension": initial["dimension"], "value": prior, "patient_id": "subject_patient"},
                             "participial_event": {"action": repair, "controlled_patient_id": "subject_patient",
                                                   "dimension": dimension, "effect": effect, "span": [participle_start, participle_end]},
                             "predicted_state": predicted,
                             "measured_result": {"tokens": result, "patient_id": "subject_patient", "dimension": stated_dimension,
                                                 "comparison": relation, "satisfied": relation_true},
                             "causal_witness": {"same_patient": True, "same_property": same_dimension,
                                                "material_support": material_support, "result_follows": relation_true and same_dimension},
                             "semantic_relation_valid": material_support and same_dimension and relation_true})
    return analyses


def render_final(words):
    result = list(words)
    parses = parse_final(words)
    if parses:
        begin, end = parses[0]["participial_event"]["span"]
        result[begin - 1] += ","
        result[end - 1] += ","
    # This spelling supplies no extra letters and is not counted as a live
    # token shift unless its independently declared boundary is crossed.
    return " ".join(result).replace("air tight", "air-tight").capitalize() + "."
