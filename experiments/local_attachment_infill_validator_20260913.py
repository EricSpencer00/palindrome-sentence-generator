"""Independent lexical and whole-sentence parser for a possessive attachment.

This module does not import the source frames, their inventories, the infill
interface, or a proposal's claimed parse.  It licenses a small English grammar
and checks the repair event and possessive attachment structurally.  It is not
a reader-quality or provenance certificate.
"""
from __future__ import annotations


def attachment_prefix(words):
    """Incrementally parse ORIGIN? QUALIFICATION? HUMAN-PLURAL.

    Return no analyses for an invalid prefix, including tokens after a head.
    The complete noun phrase denotes an owner, not an imperative or a clause.
    """
    origins = {"local", "regional", "foreign"}
    qualifications = {"young", "retired", "skilled"}
    humans = {"artists", "painters", "curators", "collectors", "restorers",
              "sculptors", "framers", "dealers", "donors", "patrons",
              "residents", "farmers"}
    words = tuple(words)
    position = 0
    origin = qualification = None
    if position < len(words) and words[position] in origins:
        origin = words[position]
        position += 1
    if position < len(words) and words[position] in qualifications:
        qualification = words[position]
        position += 1
    if position == len(words):
        return [{"complete": False, "owner_type": "human_plural",
                 "origin": origin, "qualification": qualification}]
    if position + 1 != len(words) or words[position] not in humans:
        return []
    return [{"complete": True, "owner_type": "human_plural", "head": words[position],
             "origin": origin, "qualification": qualification}]


def parse_sentence(words):
    """Consume one complete sentence without a hole position supplied by caller."""
    words = tuple(words)
    i = 0

    def take(options):
        nonlocal i
        if i >= len(words) or words[i] not in options:
            raise ValueError("grammar mismatch")
        value = words[i]
        i += 1
        return value

    try:
        actor = take({"traders", "dealers", "conservators"})
        relative = None
        if i < len(words) and words[i] == "who":
            take({"who"})
            inspection = take({"inspect", "examine"})
            observed_defect = take({"damaged", "faded"})
            collection = take({"paintings", "prints"})
            take({"borrowed"})
            take({"from"})
            region = take({"regional", "local"})
            lender = take({"museums", "galleries"})
            relative = {"agent_id": "actor", "event": inspection,
                        "patient_id": "inspected_collection", "patient": collection,
                        "observed_defect": observed_defect,
                        "loan_source": {"head": lender, "region": region}}
        manner = None
        if i < len(words) and words[i] in {"carefully", "patiently"}:
            manner = take({"carefully", "patiently"})
        action = take({"repair", "mend", "worsen"})
        scale = take({"small", "large"})
        defect = take({"tears", "rips"})
        take({"in"})
        # A bounded NP is discovered from syntax, not from a caller-provided cut.
        attachment_start = i
        candidates = []
        for width in (1, 2, 3):
            end = attachment_start + width
            if end + 2 != len(words):
                continue
            owner_parses = [p for p in attachment_prefix(words[attachment_start:end]) if p["complete"]]
            if not owner_parses or words[end] not in {"red", "blue", "green"} or words[end + 1] not in {"art", "artwork"}:
                continue
            effect = {"repair": -1, "mend": -1, "worsen": 1}[action]
            for owner in owner_parses:
                attachment = {"relation": "possesses", "owner_id": "owner",
                              "owner": owner, "patient_id": "artwork",
                              "surface_span": [attachment_start, end]}
                witness = {"action": action, "defect": defect, "initial_defect": 2,
                           "predicted_defect": 2 + effect,
                           "repair_patient_id": "artwork",
                           "defect_bearer_id": "artwork", "owner_patient_id": "artwork",
                           "valid": effect < 0 and owner["owner_type"] == "human_plural"}
                candidates.append({"complete": True, "actor": actor, "actor_id": "actor",
                                   "relative": relative, "manner": manner, "action": action,
                                   "defect": defect, "scale": scale, "patient": words[end + 1],
                                   "color": words[end], "attachment": attachment,
                                   "semantic_witness": witness,
                                   "semantic_relation_valid": witness["valid"]})
        return candidates
    except ValueError:
        return []


def render_sentence(words):
    """Add a plural possessive apostrophe at the independently parsed NP head."""
    words = tuple(words)
    parses = parse_sentence(words)
    rendered = list(words)
    if parses:
        head = parses[0]["attachment"]["surface_span"][1] - 1
        rendered[head] += "'"
    return (" ".join(rendered)).capitalize() + "."
