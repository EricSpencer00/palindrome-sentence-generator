"""Independent full parse of an owned-artwork coordination, with reference binding.

The first conjunct introduces one artwork collection and its human possessor.
The second conjunct uses the mass demonstrative ``this art`` for that same
collection.  No argument positions, source parse, or ownership claims are
accepted from the generator.  The grammar does not establish reader quality.
"""
from __future__ import annotations


def parse_sentence(words):
    words = tuple(words)
    position = 0

    def take(vocabulary):
        nonlocal position
        if position >= len(words) or words[position] not in vocabulary:
            raise ValueError("syntax mismatch")
        word = words[position]
        position += 1
        return word

    try:
        actor = take({"traders", "dealers", "conservators"})
        arrangement = take({"sort", "arrange"})
        owner_start = position
        origin = qualification = None
        if position < len(words) and words[position] in {"local", "foreign"}:
            origin = take({"local", "foreign"})
        if position < len(words) and words[position] in {"young", "retired", "skilled"}:
            qualification = take({"young", "retired", "skilled"})
        owner = take({"artists", "painters", "collectors", "curators", "dealers", "patrons"})
        owner_end = position
        observed = take({"damaged", "torn"})
        collection = take({"paintings", "prints", "posters"})
        # A source PP adds a building referent, never a competing artwork.
        source = None
        if position < len(words) and words[position] == "that":
            take({"that"})
            take({"arrived"})
            take({"from"})
            region = take({"regional", "local"})
            building = take({"museums", "galleries"})
            source = {"type": "institution", "head": building, "region": region}
        take({"and"})
        repair = take({"repair", "mend", "worsen"})
        scale = take({"small", "large"})
        defect = take({"tears", "rips"})
        take({"in"})
        determiner = take({"this", "their"})
        take({"high"})
        take({"gloss"})
        color = take({"red", "blue"})
        art = take({"art", "artwork"})
        if position != len(words):
            return []
    except ValueError:
        return []
    accessible = [{"id": "actor", "type": "human_plural"},
                  {"id": "owner", "type": "human_plural"},
                  {"id": "owned_collection", "type": "artwork_collection"}]
    if source:
        accessible.append({"id": "source_building", "type": "institution"})
    # Do not resolve ambiguous plural 'their' by assertion.  The mass
    # demonstrative has exactly one accessible artwork-compatible antecedent.
    antecedents = [r["id"] for r in accessible if r["type"] == "artwork_collection"] if determiner == "this" else []
    reference_valid = len(antecedents) == 1
    referent = antecedents[0] if reference_valid else None
    effect = {"repair": -1, "mend": -1, "worsen": 1}[repair]
    witness = {"initial_damage": 2, "predicted_damage": 2 + effect,
               "repair_defect_bearer": referent, "possessive_patient": "owned_collection",
               "same_owned_patient": referent == "owned_collection",
               "valid": reference_valid and effect < 0 and referent == "owned_collection"}
    return [{"complete": True, "actor": actor,
             "first_event": {"predicate": arrangement, "agent": "actor", "patient": "owned_collection"},
             "ownership": {"owner": owner, "origin": origin, "qualification": qualification,
                           "owner_id": "owner", "patient_id": "owned_collection",
                           "surface_span": [owner_start, owner_end]},
             "collection": {"head": collection, "observed_condition": observed, "source": source},
             "second_event": {"predicate": repair, "agent": "actor", "defect": defect,
                              "scale": scale, "patient": referent},
             "reference": {"determiner": determiner, "head": art, "color": color,
                           "accessible_referents": accessible, "compatible_antecedents": antecedents,
                           "resolved": referent, "valid": reference_valid},
             "semantic_witness": witness, "semantic_relation_valid": witness["valid"]}]


def render_sentence(words):
    words = tuple(words)
    parsed = parse_sentence(words)
    result = list(words)
    if parsed:
        result[parsed[0]["ownership"]["surface_span"][1] - 1] += "'"
    text = " ".join(result).replace("high gloss", "high-gloss")
    return text.capitalize() + "."
