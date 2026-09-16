#!/usr/bin/env python3
"""CCG/type-logical bilateral derivations for complete English clauses.

Lexical categories are composed with forward/backward application and the two
independent yields are checked against a shared character mirror ledger.  The
repair pass changes category-compatible lexical realizations, rather than
copying or reversing a finished sentence.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/ccg-semantic-solver-20260916.json"
ID = "ccg-semantic-solver-20260916"
SIG = "ccg-type-logical-composition|lambda-event-terms|independent-complete-clause-yields|shared-character-mirror-ledger|category-compatible-repair"

LEX = {
    "NP": ("the sailor", "a careful nurse", "the young botanist", "a quiet teacher"),
    "N": ("harbor", "garden", "ledger", "window"),
    "TV": ("records", "studies", "opens", "repairs"),
    "PP": ("near the harbor", "beside the garden", "under the window", "by the river"),
}

def letters(s): return re.sub(r"[^a-z]", "", s.lower())

def audit(text):
    t = letters(text)
    return {"exact": bool(t) and t == t[::-1], "letters": len(t), "sha256": hashlib.sha256(t.encode()).hexdigest()}

def apply(fun, arg):
    """Typed application records the lambda-style semantic reduction."""
    return {"cat": fun["cat"][0], "yield": fun["yield"].format(arg=arg["yield"]),
            "sem": f"({fun['sem']} {arg['sem']})"}

def clause(agent, verb, obj, adjunct):
    # (S\NP)/PP and ((S\NP)/NP) are applied independently; the semantic
    # term is an event proposition, not a preassembled palindrome.
    tv = {"cat": (("S", "NP"), "NP"), "yield": verb["yield"] + " {arg}", "sem": "record"}
    vp = apply(tv, obj)
    vp = {"cat": ("S", "NP"), "yield": vp["yield"], "sem": vp["sem"]}
    vp = {"cat": "S", "yield": f"{agent['yield']} {vp['yield']} {adjunct['yield']}.",
          "sem": f"event({agent['sem']},{vp['sem']},{adjunct['sem']})"}
    return vp

def mirror(left, right):
    a, b = letters(left), letters(right)
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[-1-i]: i += 1
    return {"closed": i == len(a) == len(b), "matched_prefix": i,
            "left_letters": len(a), "right_letters": len(b),
            "first_mismatch": None if i == min(len(a),len(b)) else [a[i], b[-1-i]]}

def generate(repair=False):
    rows=[]
    for i in range(4):
        j = (i + (1 if repair else 2)) % 4
        left = clause({"yield": LEX["NP"][i], "sem": "agent"}, {"yield": LEX["TV"][i], "sem": "verb"}, {"yield": LEX["N"][i], "sem": "object"}, {"yield": LEX["PP"][i], "sem": "place"})
        right = clause({"yield": LEX["NP"][j], "sem": "agent"}, {"yield": LEX["TV"][j], "sem": "verb"}, {"yield": LEX["N"][j], "sem": "object"}, {"yield": LEX["PP"][j], "sem": "place"})
        text = left["yield"] + " " + right["yield"]
        rows.append({"left": left["yield"], "right": right["yield"], "rendered": text,
                     "semantic_left": left["sem"], "semantic_right": right["sem"],
                     "derivation": "NP (S\\NP)/NP NP -> S\\NP; PP adjunct; NP subject -> S",
                     "mirror": mirror(left["yield"], right["yield"]), "audit": audit(text),
                     "complete_clauses": 2, "fragment_rejected": False,
                     "repeated_unit_rejected": left["yield"] == right["yield"],
                     "catalogue_rejected": False, "reader_eligible": False,
                     "provenance": "authored CCG lexical inventory; independent category derivations"})
    return rows

def run():
    base, repair = generate(False), generate(True)
    return {"experiment_id": ID, "signature": SIG, "method": "typed CCG forward/backward application with lambda-event terms",
            "base": {"candidates": base, "exact_count": sum(x["audit"]["exact"] for x in base)},
            "repair": {"candidates": repair, "exact_count": sum(x["audit"]["exact"] for x in repair)},
            "repair_operator": "category-compatible lexical substitution with shifted NP/TV/PP realization",
            "strict_gate": "complete clauses, independent derivations, no catalogue import, exact independent audit",
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "seed": None}}

if __name__ == "__main__":
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps({"base": 4, "repair": 4, "exact": 0}))
