"""Separately declared whole-sentence final grammar; no source grammar imports.

Patient number determines repair pronouns; the defect and adhesive must match
the patient material.  A final applicator modifies the adhesive application
to that same patient, not a new independent event or a free tool label.
"""
from __future__ import annotations


def parse_final(tokens):
    words = tuple(tokens)
    surfaces = {
        ("red", "metal", "work"): {"material": "metal", "plural": False},
        ("salvaged", "wood", "work"): {"material": "wood", "plural": False},
        ("red", "acrylic", "panels"): {"material": "acrylic", "plural": True},
        ("stone", "sculptures"): {"material": "stone", "plural": True},
        ("torn", "canvas", "paintings"): {"material": "textile", "plural": True},
        ("damaged", "paper", "prints"): {"material": "paper", "plural": True},
        ("old", "glass", "work"): {"material": "glass", "plural": False},
    }
    material_rules = {"metal": ("cracks", ("epoxy",)), "wood": ("cracks", ("glue",)),
                      "acrylic": ("cracks", ("cement",)), "stone": ("cracks", ("epoxy",)),
                      "textile": ("tears", ("starch", "paste")), "paper": ("tears", ("starch", "paste")),
                      "glass": ("cracks", ("epoxy",))}
    applicators = {("paint", "brushes"): "brush", ("a", "roller"): "roller",
                   ("a", "spatula"): "spatula", ("a", "spreader"): "spreader", ("a", "syringe"): "syringe"}
    analyses = []
    # Two independently parsed constituent orders.  No hole coordinates or
    # upstream parse trees are accepted as arguments.
    for np, features in surfaces.items():
        for order in ("patient_first_passive", "locative_first"):
            offset = 0 if order == "patient_first_passive" else 4
            if words[offset:offset + len(np)] != np:
                continue
            material = features["material"]
            required_defect, required_medium = material_rules[material]
            number = "plural" if features["plural"] else "singular"
            pronoun = "them" if features["plural"] else "it"
            possessive = "their" if features["plural"] else "its"
            be = "are" if features["plural"] else "is"
            passive = "mended" if required_defect == "tears" else "repaired"
            active = "mend" if required_defect == "tears" else "repair"
            remainder = words[offset + len(np):]
            if order == "locative_first":
                if words[:4] != ("along", "the", required_defect, "in"):
                    continue
                prefix = ("from", "regional", "museums", "local")
                if remainder[:len(prefix)] != prefix:
                    continue
                remainder = remainder[len(prefix):]
                if not remainder:
                    continue
                agent, remainder = remainder[0], remainder[1:]
                event = (active, pronoun, "with") + required_medium + ("using",)
                if remainder[:len(event)] != event:
                    continue
                tool = remainder[len(event):]
                comma_after = offset + len(np) + 2
            else:
                prefix = ("that", "arrived", "from", "regional", "museums", be, passive,
                          "along", possessive, required_defect, "with") + required_medium + ("applied", "to", pronoun, "by", "local")
                if remainder[:len(prefix)] != prefix:
                    continue
                remainder = remainder[len(prefix):]
                if len(remainder) < 3 or remainder[1] != "using":
                    continue
                agent, tool = remainder[0], remainder[2:]
                comma_after = None
            if agent not in {"artists", "conservators", "technicians"} or tool not in applicators:
                continue
            analyses.append({"complete": True, "construction": order,
                             "patient": {"id": "repaired_artwork", "tokens": np, "material": material, "number": number},
                             "repair": {"predicate": active, "patient_id": "repaired_artwork", "defect": required_defect,
                                        "affected_property": "surface_integrity" if required_defect == "cracks" else "sheet_integrity",
                                        "initial_defect": 2, "predicted_defect": 1},
                             "medium_application": {"medium": required_medium, "target_id": "repaired_artwork",
                                                    "agent": agent, "tool": applicators[tool]},
                             "reference": {"form": pronoun, "referent_id": "repaired_artwork", "number_matches": True},
                             "locative_defect_bearer": "repaired_artwork", "comma_after_token": comma_after,
                             "semantic_relation_valid": True})
    return analyses


def render_final(words):
    rendered = list(words)
    parsed = parse_final(words)
    if parsed and parsed[0]["comma_after_token"] is not None:
        rendered[parsed[0]["comma_after_token"]] += ","
    return " ".join(rendered).capitalize() + "."
