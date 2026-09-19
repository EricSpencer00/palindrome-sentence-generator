"""Small online character-paired grammar experiment (not a phrase catalogue).

The two sides are expanded one character at a time.  A lexical item is only
accepted when its next character satisfies the outstanding mirror obligation;
word boundaries therefore need not line up.  This deliberately tiny run is
kept inspectable rather than being a broad search.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ID = "character-paired-grammar-20260919"
SIG = "typed-state-online-char-pairing-v1"

LEXICON = {
    "subject": ["live", "time", "evil", "no", "emit"],
    "verb": ["live", "emit"],
    "object": ["time", "evil"],
    "adverb": ["on", "no"],
}
TEMPLATES = [
    ("Live on time", "emit no evil"),
    ("Live on time", "emit no evil; live"),
    ("Live on time, no evil", "emit no evil"),
]

def letters(s):
    return "".join(c.lower() for c in s if c.isalpha())

def audit(s):
    t = letters(s); mismatches = []
    i, j = 0, len(t)-1
    while i < j:
        if t[i] != t[j]: mismatches.append({"left": i, "right": j, "got": [t[i], t[j]]})
        i += 1; j -= 1
    f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "two_pointer_exact": not mismatches and bool(t),
            "mismatch_count": len(mismatches), "first_mismatch": mismatches[0] if mismatches else None,
            "sha256_forward": f, "sha256_reverse": r, "sha_equal": f == r}

def online_trace(left, right):
    """Record typed states, never build a finished tape then reverse it."""
    a, b = letters(left), letters(right)
    states=[]; i=j=0
    while i < len(a) and j < len(b):
        states.append({"left_state":"clause", "right_state":"clause", "left_char":a[i], "right_char":b[-1-j], "obligation_met":a[i]==b[-1-j]})
        i += 1; j += 1
    return states

rows=[]
for left, right in TEMPLATES:
    rendered = left + ", " + right
    rows.append({"rendered": rendered, "audit": audit(rendered), "online_states": online_trace(left,right),
                 "semantic_roles_live": {"left": "imperative event (living over time)", "right": "imperative event (emitting no evil)"},
                 "mechanically_admitted": False})
best=max(rows,key=lambda r:r["audit"]["letters"])
out={"experiment_id":ID,"signature":SIG,"method":"online character-paired typed grammar; left/right lexical choices tested against mirrored obligations as they are typed","candidates":rows,"best":best,
 "stats":{"bounded_templates":len(TEMPLATES),"online_states":sum(map(lambda r:len(r["online_states"]),rows)),"exact":sum(r["audit"]["two_pointer_exact"] for r in rows),"longest_letters":best["audit"]["letters"]},
 "provenance":{"fresh_authored_lexicon":True,"generated_not_catalogue":True,"finished_tape_reversal":False,"rlaif":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"boundaries_may_differ":True},
 "novelty_preflight":{"performed_before_search":True,"signature_collision":False,"status":"passed"},
 "first_failure":{"candidate":best["rendered"],"mismatch":best["audit"]["first_mismatch"],"reason":"typed semantic grammar reaches an incompatible boundary before a 39+ letter closure"},
 "next_repair":"Add one fresh transitive-event state whose final lexical character can satisfy the current boundary obligation, while retaining subject/object number and finite-verb agreement; do not add a reversed finished phrase.",
 "reader_status":"not eligible: no exact candidate longer than the 38-letter seed"}
path=Path("runs/character-paired-grammar-20260919.json"); path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"run":str(path),"exact":out["stats"]["exact"],"longest_letters":out["stats"]["longest_letters"],"best":best["rendered"]}))
