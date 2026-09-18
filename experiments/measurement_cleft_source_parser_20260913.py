"""Independent source grammar for equative scalar/repair clefts."""
from __future__ import annotations


def parse_source(tokens):
    words = tuple(tokens)
    discourse = words[:1] == ("nonetheless",)
    start = int(discourse)
    roles = {
        "height": {"heads": {"sections", "sectors"}, "feature": ("ridges",), "measure": ("ridge", "height"),
                   "comparison": "lower", "before": 3, "process": ("coarse", "sandpaper"),
                   "headed_actions": {"reduces", "cuts", "raises"}, "scalar_actions": {"reduces", "raises"}},
        "roughness": {"heads": {"surfaces", "patches"}, "feature": ("areas",), "measure": ("surface", "roughness"),
                      "comparison": "lower", "before": 3, "process": ("fine", "sandpaper"),
                      "headed_actions": {"smooths", "roughens"}, "scalar_actions": {"reduces", "raises"}},
        "firmness": {"heads": {"joints", "seams"}, "feature": ("connections",), "measure": ("joint", "firmness"),
                     "comparison": "higher", "before": 1, "process": ("fresh", "adhesive"),
                     "headed_actions": {"secures", "weakens"}, "scalar_actions": {"increases", "reduces"}},
        "permeability": {"heads": {"slabs", "panels"}, "feature": ("plates",), "measure": ("surface", "permeability"),
                         "comparison": "lower", "before": 2, "process": ("fresh", "adhesive"),
                         "headed_actions": {"seals", "opens"}, "scalar_actions": {"reduces", "raises"}},
    }
    materials = {("red", "metalwork"): "metal", ("old", "woodwork"): "wood", ("clear", "glasswork"): "glass"}
    effects = {"reduces": -2, "cuts": -2, "raises": 2, "smooths": -2, "roughens": 2,
               "secures": 2, "weakens": -1, "increases": 2, "seals": -2, "opens": 2}
    analyses = []
    for metric, role in roles.items():
        for mode in ("scalar_what_cleft", "headed_relative_equative"):
            if mode == "scalar_what_cleft":
                initial = ("what", "is", role["comparison"])
                if words[start:start + len(initial)] != initial:
                    continue
                copula, identity, allowed = "is", role["measure"], role["scalar_actions"]
                head = "what"
            else:
                if start >= len(words) or words[start] not in role["heads"]:
                    continue
                head = words[start]
                initial = (head, "whose", metric, "is", role["comparison"])
                if words[start:start + len(initial)] != initial:
                    continue
                copula, identity, allowed = "are", role["feature"], role["headed_actions"]
            position = start + len(initial)
            baseline = ("than", "before", "the", "recent", "repair", "on", "the")
            if words[position:position + len(baseline)] != baseline:
                continue
            position += len(baseline)
            for material_np, material in materials.items():
                body = material_np + (copula, "the") + identity + ("that",) + role["process"]
                if words[position:position + len(body)] != body or position + len(body) + 1 != len(words):
                    continue
                action = words[-1]
                if action not in allowed:
                    continue
                before, delta = role["before"], effects[action]
                after = max(0, before + delta)
                comparison_true = after < before if role["comparison"] == "lower" else after > before
                analyses.append({"complete": True, "mode": mode, "concessive": discourse,
                    "relative_head": head, "material": material, "patient_id": "measured_feature",
                    "measure": {"property": metric, "patient_id": "measured_feature", "before": before, "after": after},
                    "equative": {"identity_tokens": identity, "patient_id": "measured_feature", "measure_property": metric},
                    "repair_relative": {"finite_predicate": action, "process": role["process"], "patient_id": "measured_feature",
                                        "affected_property": metric, "effect": delta},
                    "comparison": role["comparison"], "semantic_relation_valid": comparison_true})
    return analyses
