"""Target analysis: compositional credential NP and owner-bound lexical role."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    position: str
    pending_head: str = ""
    predicate: str = ""
    sponsor: str = ""
    role_domain: str = ""


def target_steps(s):
    def arc(word, position, pending=None, predicate=None, sponsor=None, role=None):
        return word, Target(position, s.pending_head if pending is None else pending,
                            s.predicate if predicate is None else predicate,
                            s.sponsor if sponsor is None else sponsor, s.role_domain if role is None else role)
    verbs = ("specify", "name", "mark", "reveal", "distinguish", "describe", "label", "identify")
    if s.position == "initial":
        return [arc(a, "credential_head", pending=b) for a, b in (("key", "cards"), ("book", "plates"), ("arm", "bands"), ("wrist", "bands"), ("pass", "cards"), ("number", "plates"))]
    if s.position == "credential_head":
        return [arc(s.pending_head, "credential", pending="")]
    if s.position == "credential":
        return [arc("that", "display_relation")] + [arc(w, "definite_object", predicate=w) for w in verbs]
    if s.position == "display_relation":
        return [arc("display", "identity_modifier")]
    if s.position == "identity_modifier":
        return [arc("identifying", "information_head")]
    if s.position == "information_head":
        return [arc("information", "passive_supplier")]
    if s.position == "passive_supplier":
        return [arc("supplied", "supplier_preposition")]
    if s.position == "supplier_preposition":
        return [arc("by", "supplier_modifier")]
    if s.position == "supplier_modifier":
        return [arc("authorized", "supplier_head")]
    if s.position == "supplier_head":
        return [arc("officials", "main_predicate")]
    if s.position == "main_predicate":
        return [arc(w, "definite_object", predicate=w) for w in verbs]
    if s.position == "definite_object":
        return [arc("the", "sponsor")]
    if s.position == "sponsor":
        return [arc(w, "role", sponsor=w) for w in ("smiths", "bishops", "locals", "residents", "workers", "guards")]
    if s.position == "role":
        return [arc(word, "finished", role=domain) for word, domain in (("watchman", "watch"), ("boatman", "boat"), ("doorman", "door"))]
    return []


def target_meaning(s):
    if s.position != "finished":
        return None
    return {"shared_core": ("credential", s.predicate, "duty_attendant", s.role_domain, s.sponsor),
            "attachment": {"possessor": s.sponsor, "attaches_to": "attendant", "role_domain": s.role_domain},
            "constituency": "[owners' lexical-role]"}
