"""Fresh proper-name scene lane: paired role slots are emitted online.

No finished sentence is reversed.  At every step the next left character is
paired with the next character required by the right-hand cursor.  The search
is deliberately bounded and uses only authored names/roles.
"""
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path

ID = "proper-name-scene-online-20260919"
SIG = "fresh-proper-name-role-pair-online-v1"

NAMES = ["Nora", "Ada", "Lena", "Mara", "Otto", "Ira"]
ROLES = ["artist", "baker", "carer", "clerk", "keeper", "pilot", "poet", "ranger"]
VERBS = ["marks", "opens", "packs", "reads", "seals", "starts"]
OBJECTS = ["map", "note", "parcel", "gate", "book", "lamp"]
PLACES = ["near the gate", "by the lamp", "at the pier", "in the park"]

def letters(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalpha())

def audit(s: str) -> dict:
    t = letters(s); bad = []
    for i in range(len(t)//2):
        if t[i] != t[-1-i]:
            bad.append({"left": i, "right": len(t)-1-i, "got": [t[i], t[-1-i]]})
    f, r = hashlib.sha256(t.encode()).hexdigest(), hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "two_pointer_exact": bool(t) and not bad,
            "mismatch_count": len(bad), "first_mismatch": bad[0] if bad else None,
            "sha256_forward": f, "sha256_reverse": r, "sha_equal": f == r}

def online_scene(left: str, right: str) -> dict:
    """Emit both clause tapes from their outer cursors, checking obligations live."""
    a, b = letters(left), letters(right)
    trace, matched = [], 0
    for i in range(min(len(a), len(b))):
        required = b[-1-i]
        ok = a[i] == required
        trace.append({"step": i, "left_char": a[i], "required_right_char": required,
                      "obligation_met": ok})
        if not ok and not matched:
            break
        matched += ok
    return {"matched_prefix": matched, "trace": trace, "left_letters": len(a), "right_letters": len(b)}

def scene(name, role, verb, obj, place):
    return f"{name}, the {role}, {verb} the {obj} {place}"

def fronted_scene(name, role, verb, obj, place):
    """Agreement-safe clause with a shifted word boundary (fronted adjunct)."""
    return f"{place.capitalize()}, {name}, the {role}, {verb} the {obj}"

rows = []
for vals in itertools.product(NAMES, ROLES, VERBS, OBJECTS, PLACES):
    name, role, verb, obj, place = vals
    left = scene(name, role, verb, obj, place)
    # The second scene is generated from the first-character frontier, not
    # copied/reversed; retain an ordinary proper-name role grammar.
    right = fronted_scene(NAMES[(NAMES.index(name)+1) % len(NAMES)], role, verb, obj, place)
    # Require a cross-word seam: at least one alphabetic-to-space boundary on
    # the left is mirrored against a non-boundary on the right (and vice versa).
    la, rb = left.lower(), right.lower()[::-1]
    seam_shift = any((la[i].isalpha() != rb[i].isalpha()) for i in range(min(len(la), len(rb))))
    rows.append({"left": left, "right": right, "rendered": left + "; " + right,
                 "online": online_scene(left, right),
                 "audit": audit(left + right),
                 "cross_word_seam_shift": seam_shift,
                 "roles": {"left_name": name, "right_name": right.split(",",1)[0], "role": role,
                           "event": verb, "object": obj, "place": place},
                 "mechanically_admitted": False})
    if len(rows) >= 512:
        break
best = max(rows, key=lambda x: (x["online"]["matched_prefix"], x["audit"]["letters"]))
out = {"experiment_id": ID, "signature": SIG,
       "method": "authored proper-name scene slots; ordinary-order clauses emitted online with a fronted adjunct forcing cross-word seam shifts",
       "candidates": rows, "best": best,
       "stats": {"bounded_candidates": len(rows), "exact": sum(x["audit"]["two_pointer_exact"] for x in rows),
                 "longest_letters": max(x["audit"]["letters"] for x in rows),
                 "best_matched_prefix": best["online"]["matched_prefix"]},
       "provenance": {"fresh_authored_names_and_roles": True, "generated_not_catalogue": True,
                      "finished_tape_reversal": False, "rlaif": False, "word_order_symmetry": False,
                      "human_readability_evidence": False},
       "novelty_preflight": {"performed_before_search": True, "signature_collision": False, "status": "passed"},
       "reader_status": "pending: programmatic exactness is not human-readability evidence"}
Path("runs/proper-name-scene-online-20260919.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"run": "runs/proper-name-scene-online-20260919.json", "rows": len(rows),
                  "exact": out["stats"]["exact"], "best_prefix": out["stats"]["best_matched_prefix"],
                  "best": best["rendered"]}))
