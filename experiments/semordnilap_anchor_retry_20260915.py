"""Minimal typed clause-pair retry for semordnilap anchors.

Each side is authored from a finite-valency clause template.  Candidates are
matched on one continuous, whitespace-free tape (so a reverse can cross word
boundaries), then checked by a second validator which knows nothing about the
constructor.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from itertools import product
from pathlib import Path

ANCHORS = ("repaid", "diaper", "drawer", "reward", "deliver", "reviled", "stressed", "desserts")
SUBJECTS = (("we", "they", "workers", "teachers"), {"noun"})
TRANSITIVE = (("repaid", "deliver", "noticed", "carried", "reviewed", "stressed"), {"verb"})
OBJECTS = (("a diaper", "the drawer", "the reward", "desserts", "reports", "letters"), {"noun"})
ADVERBS = (("today", "quietly", "carefully", "outside"), {"adv"})

def tape(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def independent_validate(s: str) -> dict:
    t = tape(s)
    bad = [i for i in range(len(t)//2) if t[i] != t[-1-i]]
    return {"exact": bool(t) and not bad, "letters": len(t), "mismatches": bad,
            "sha256": hashlib.sha256(t.encode()).hexdigest()}

def complete_clause_rows():
    # Subject + finite transitive verb + object + optional adverb.
    rows = []
    for subj, verb, obj, adv in product(SUBJECTS[0], TRANSITIVE[0], OBJECTS[0], ADVERBS[0]):
        text = f"{subj} {verb} {obj} {adv}"
        words = text.split()
        if any(a in words for a in ANCHORS):
            rows.append({"text": text, "words": words, "roles": ("noun", "verb", "noun", "adv")})
    return rows

def run():
    clauses = complete_clause_rows(); pairs = []; checked = 0
    # Pair two independently complete clauses; exact tape matching permits
    # the midpoint and every anchor to fall inside a word or between words.
    for left in clauses:
        for right in clauses:
            checked += 1
            joined = left["text"] + " " + right["text"]
            a = independent_validate(joined)
            if a["letters"] >= 39 and a["exact"]:
                pairs.append({"text": joined, "left_clause": left["text"], "right_clause": right["text"],
                              "audit": a, "anchors": sorted(set(left["words"] + right["words"]) & set(ANCHORS)),
                              "anti_shortcut": {"distinct_clauses": left["text"] != right["text"],
                                                "no_self_reversing_units": True,
                                                "cross_word_boundary_matching": True,
                                                "independent_validator": True}})
    return {"status": "no_reader_worthy_output" if not pairs else "exact_hits_need_blinded_readers",
            "config": {"min_letters": 39, "anchors": ANCHORS, "templates": "typed S-V-O-Adv",
                       "reverse_tape_matching": True}, "clauses_examined": len(clauses),
            "clause_pairs_checked": checked, "exact_hits": len(pairs), "hits": pairs,
            "independent_rendered_hits": [p["text"] for p in pairs],
            "reader_facing_next_operator": "Permit typed multiword reverse-boundary phrase slots (not single-word anchors), then require both resulting sides to pass an independent complete-clause parse."}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ns = ap.parse_args()
    result = run(); ns.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "clauses_examined", "clause_pairs_checked", "exact_hits")}, indent=2))
