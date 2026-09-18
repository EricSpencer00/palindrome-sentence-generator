"""Independent source-side agentless repair/result semantic grammar."""
from __future__ import annotations


def parse_source(tokens):
    words = tuple(tokens)
    starts = {("rough",): ("roughness", 3, False), ("leaky",): ("leakage", 2, False),
              ("loose", "joints", "in"): ("firmness", 1, True),
              ("reworked", "ridges", "on"): ("height", 3, True)}
    materials = {("metalwork",): ("metal", False), ("woodwork",): ("wood", False),
                 ("glasswork",): ("glass", False), ("oak", "panels"): ("wood", True)}
    effects = {"polished": ("roughness", -2), "sealed": ("leakage", -2),
               "tightened": ("firmness", 2), "sanded": ("height", -2),
               "loosened": ("firmness", -1), "roughened": ("roughness", 2)}
    means = {"polished": ("fine", "abrasives"), "sealed": ("fresh", "adhesive"),
             "tightened": ("steel", "screws"), "sanded": ("coarse", "abrasives"),
             "loosened": ("steel", "screws"), "roughened": ("fine", "abrasives")}
    judgments = {("smoother",): ("roughness", "decreased"), ("smoother", "than", "before"): ("roughness", "decreased"),
                 ("airtight",): ("leakage", "zero"), ("less", "permeable"): ("leakage", "decreased"),
                 ("firmer",): ("firmness", "increased"), ("more", "secure"): ("firmness", "increased"),
                 ("lower",): ("height", "decreased"), ("lower", "than", "before"): ("height", "decreased")}
    rows = []
    for initial, (prop, before, plural_head) in starts.items():
        if words[:len(initial)] != initial:
            continue
        for np, (material, plural_material) in materials.items():
            i = len(initial)
            if words[i:i + len(np)] != np:
                continue
            i += len(np)
            relative = ("that", "arrived", "from", "regional", "museums")
            if words[i:i + len(relative)] != relative:
                continue
            i += len(relative)
            if i >= len(words) or words[i] not in effects:
                continue
            action = words[i]
            i += 1
            modifier = ("with",) + means[action] + ("under", "steady", "pressure")
            if words[i:i + len(modifier)] != modifier:
                continue
            i += len(modifier)
            plural = plural_head or plural_material
            if words[i:i + 2] != (("are" if plural else "is"), "now"):
                continue
            ending = words[i + 2:]
            if ending not in judgments:
                continue
            action_property, delta = effects[action]
            result_property, comparison = judgments[ending]
            after = max(0, before + delta) if action_property == prop else before
            result_true = {"zero": after == 0, "decreased": after < before, "increased": after > before}[comparison]
            support = not (prop == "firmness" and material == "glass")
            valid = support and action_property == prop == result_property and result_true
            rows.append({"complete": True, "patient_id": "affected_patient", "host_material": material,
                         "property": prop, "initial_state": before, "repair": {"action": action,
                         "patient_id": "affected_patient", "property": action_property, "delta": delta},
                         "predicted_state": after, "result": {"tokens": ending, "patient_id": "affected_patient",
                         "property": result_property, "comparison": comparison, "true_in_predicted_state": result_true},
                         "number": "plural" if plural else "singular", "semantic_relation_valid": valid})
    return rows
