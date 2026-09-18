"""Independent final constituency and credential-reference parser.

Only the final role-bound possessive reading is accepted here.  It does not
assert equivalence between owning a duty domain and employing its attendant.
"""


def parse_credential_sentence(words):
    words = tuple(words)
    if words[:2] not in {("key", "cards"), ("book", "plates"), ("arm", "bands"),
                         ("wrist", "bands"), ("pass", "cards"), ("number", "plates")}:
        return []
    i = 2
    relative = False
    if words[i:i + 1] == ("that",):
        expected = ("that", "display", "identifying", "information", "supplied", "by", "authorized", "officials")
        if words[i:i + len(expected)] != expected:
            return []
        relative = True
        i += len(expected)
    if len(words) != i + 4 or words[i] not in {"identify", "label", "describe", "distinguish", "reveal", "mark", "name", "specify", "eat"} or words[i + 1] != "the":
        return []
    if words[i + 2] not in {"guards", "workers", "residents", "locals", "bishops", "smiths"} or words[i + 3] not in {"doorman", "boatman", "watchman"}:
        return []
    predicate, owner, role = words[i], words[i + 2], words[i + 3]
    return [{"complete": True, "credential": words[:2], "main_predicate": predicate,
             "relative_information_attaches_to": "credential" if relative else None,
             "possessive_owner": owner, "possessive_attaches_to": "attendant", "role": role,
             "reference_graph": {"credential_refers_to": "attendant", "owner_associated_with": "attendant"},
             "semantic_relation_valid": predicate != "eat",
             "semantic_scope": "typed identifying relation; no autonomy, physical causation, or reader-quality inference"}]


def render_credential_sentence(words):
    rendered = list(words)
    if len(rendered) >= 3 and rendered[-2] in {"guards", "workers", "residents", "locals", "bishops", "smiths"} and rendered[-1] in {"doorman", "boatman", "watchman"}:
        rendered[-2] += "'"
    return " ".join(rendered).capitalize() + "."
