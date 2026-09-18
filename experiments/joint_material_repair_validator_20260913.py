"""Independent typed parser for two events on one owned physical artwork.

No source inventories or source analyses are imported.  The material limits
which first action, final finish and damaged property make physical sense.
The demonstrative binds the second event to the unique owned artwork.
"""
from __future__ import annotations


def parse_sentence(tokens):
    words = tuple(tokens)
    i = 0

    def take(allowed):
        nonlocal i
        if i >= len(words) or words[i] not in allowed:
            raise ValueError("syntax mismatch")
        word = words[i]
        i += 1
        return word

    try:
        if words[:2] in {("metal", "workers"), ("wood", "workers")}:
            actor = words[:2]
            i = 2
        else:
            actor = (take({"traders", "conservators", "metalworkers", "woodworkers"}),)
        first = take({"sort", "solder", "seal", "brush"})
        first_property = {"sort": "arrangement", "solder": "seam_continuity",
                          "seal": "permeability", "brush": "surface_dust"}[first]
        if first == "solder":
            take({"seams"})
            take({"in"})
        elif first == "seal":
            take({"pores"})
            take({"in"})
        elif first == "brush":
            take({"dust"})
            take({"from"})
        owner_start = i
        take({"local"})
        owner = take({"artists", "collectors", "patrons"})
        owner_end = i
        take({"damaged"})
        material_word = take({"metal", "oak", "canvas"})
        head = take({"panels", "paintings"})
        materials = {("metal", "panels"): "metal", ("oak", "panels"): "rigid_wood",
                     ("canvas", "paintings"): "textile"}
        material = materials.get((material_word, head))
        take({"that"})
        take({"arrived"})
        take({"from"})
        take({"regional"})
        take({"museums"})
        take({"and"})
        repair = take({"repair", "mend", "worsen"})
        defect = take({"cracks", "tears"})
        take({"in"})
        reference = take({"this", "their"})
        finish_start = i
        finish = None
        substrate = None
        if words[i:i + 2] == ("high", "gloss"):
            i += 2
            finish = "glossy_coating"
            take({"red"})
        elif i < len(words) and words[i] == "matte":
            i += 1
            finish = "matte_coating"
            take({"red"})
        elif words[i:i + 2] == ("polished", "red"):
            i += 2
            finish = "polished_surface"
            if words[i:i + 2] == ("hard", "wood"):
                i += 2
            else:
                take({"hardwood"})
            substrate = "rigid_wood"
        else:
            return []
        finish_end = i
        take({"art"})
        if i != len(words):
            return []
    except (ValueError, IndexError):
        return []
    initial_support = {"sort": {"metal", "rigid_wood", "textile"},
                       "solder": {"metal"}, "seal": {"rigid_wood", "textile"},
                       "brush": {"metal", "rigid_wood", "textile"}}
    defect_support = {"cracks": {"metal", "rigid_wood"}, "tears": {"textile"}}
    repair_support = {"repair": {"cracks"}, "mend": {"tears"}, "worsen": set()}
    # Finish is a property of the same physical surface, not a second artwork.
    finish_compatible = material is not None and (substrate is None or substrate == material)
    reference_valid = reference == "this"
    repaired_property = "surface_continuity" if defect == "cracks" else "textile_continuity"
    supported = (material in initial_support[first] and material in defect_support[defect]
                 and defect in repair_support[repair] and finish_compatible and reference_valid)
    return [{"complete": True, "actor": actor, "ownership": {"owner": owner,
             "span": [owner_start, owner_end], "patient_id": "owned_artwork"},
             "material": material, "source_material_tokens": [material_word, head],
             "first_event": {"action": first, "affected_property": first_property,
                             "patient_id": "owned_artwork", "material_compatible": material in initial_support[first]},
             "repair_event": {"action": repair, "defect": defect, "affected_property": repaired_property,
                              "patient_id": "owned_artwork" if reference_valid else None,
                              "initial_defect": 2, "predicted_defect": 1 if supported else 2},
             "final_property": {"finish": finish, "substrate_constraint": substrate,
                                "span": [finish_start, finish_end], "bearer_id": "owned_artwork" if reference_valid else None,
                                "material_compatible": finish_compatible},
             "reference": {"surface": reference, "compatible_artwork_antecedents": ["owned_artwork"],
                           "resolved": "owned_artwork" if reference_valid else None, "valid": reference_valid},
             "semantic_relation_valid": supported}]


def render_sentence(words):
    result = list(words)
    parsed = parse_sentence(words)
    if parsed:
        result[parsed[0]["ownership"]["span"][1] - 1] += "'"
    return " ".join(result).replace("high gloss", "high-gloss").capitalize() + "."
