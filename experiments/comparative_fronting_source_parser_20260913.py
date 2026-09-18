"""Independent source parser for comparative inversion after a repair process."""
from __future__ import annotations


def parse_source(tokens):
    words = tuple(tokens)
    discourse = words[:1] == ("nonetheless",)
    start = int(discourse)
    comparisons = {("smoother",): ("roughness", "less"), ("less", "permeable"): ("leakage", "less"),
                   ("firmer",): ("firmness", "more"), ("more", "secure"): ("firmness", "more"),
                   ("lower",): ("height", "less")}
    actions = {"polishing": ("roughness", -2, ("fine", "abrasives")),
               "sealing": ("leakage", -2, ("fresh", "adhesive")),
               "tightening": ("firmness", 2, ("steel", "screws")),
               "sanding": ("height", -2, ("coarse", "abrasives")),
               "loosening": ("firmness", -1, ("steel", "screws"))}
    features = {("rough",): ("roughness", 3, False), ("leaky",): ("leakage", 2, False),
                ("loose", "joints", "in"): ("firmness", 1, True),
                ("reworked", "ridges", "on"): ("height", 3, True)}
    materials = {("metalwork",): ("metal", False), ("woodwork",): ("wood", False),
                 ("glasswork",): ("glass", False), ("oak", "panels"): ("wood", True),
                 ("wooden", "vessel"): ("wood", False)}
    analyses = []
    for comparative, (stated_property, direction) in comparisons.items():
        initial = comparative + ("than", "before", "after")
        if words[start:start + len(initial)] != initial:
            continue
        pos = start + len(initial)
        if pos >= len(words) or words[pos] not in actions:
            continue
        action = words[pos]
        affected, delta, applicator = actions[action]
        process = ("with",) + applicator + ("under", "steady", "pressure", "for", "several", "hours")
        pos += 1
        if words[pos:pos + len(process)] != process:
            continue
        pos += len(process)
        if words[pos:pos + 1] not in {("is",), ("are",)} or words[pos + 1:pos + 2] != ("the",):
            continue
        copula = words[pos]
        subject = words[pos + 2:]
        for first, (prop, before, feature_plural) in features.items():
            for material_tokens, (material, material_plural) in materials.items():
                if subject != first + material_tokens:
                    continue
                plural = feature_plural or material_plural
                if copula != ("are" if plural else "is"):
                    continue
                after = max(0, before + delta) if affected == prop else before
                comparison_true = after < before if direction == "less" else after > before
                same_property = prop == affected == stated_property
                physical = not (prop == "firmness" and material == "glass")
                analyses.append({"complete": True, "construction": "comparative_fronting",
                    "concessive": discourse, "patient_id": "postposed_patient", "patient_tokens": subject,
                    "material": material, "initial_property": prop, "before": before,
                    "repair": {"gerund": action, "patient_id": "postposed_patient", "property": affected, "delta": delta},
                    "after": after, "comparison": {"tokens": comparative, "patient_id": "postposed_patient",
                    "property": stated_property, "direction": direction, "true": comparison_true},
                    "semantic_relation_valid": same_property and comparison_true and physical})
    return analyses
