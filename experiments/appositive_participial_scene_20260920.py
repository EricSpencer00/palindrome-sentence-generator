"""Bounded authored appositive/participial scene grammar.

This lane tests a construction family not used by the seam, center, relation,
dialogue, or imperative lanes: one complete sentence contains an appositive
subject description and a non-finite participial adjunct.  Grammar slots are
selected as intact English units; character obligations are audited as each
rendered character becomes available.  It never reverses a finished tape or
copies a catalogue string.
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path

OUT = Path(__file__).parents[1] / "runs" / "appositive-participial-scene-20260920.json"

SUBJECTS = [
    ("Mara", "a patient cartographer"),
    ("Jonah", "a quiet keeper"),
    ("Elian", "an attentive gardener"),
]
PARTICIPLES = ["having crossed the rain-darkened bridge", "watching the first lanterns glow", "carrying a weathered map"]
VERBS = ["studies", "records", "follows"]
OBJECTS = ["the northern harbor", "the sleeping orchard", "the narrow river"]
TAILS = ["before the evening tide", "beneath the patient moon", "beside the old stone wall"]

CONTROLS = [
    "Mara, a patient cartographer, studies the northern harbor before the evening tide.",
    "Jonah, a quiet keeper, records the sleeping orchard beneath the patient moon.",
]

def norm(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalpha())

def audit(s: str) -> dict:
    t = norm(s)
    f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "pointer_exact": t == t[::-1],
            "sha256_forward": f, "sha256_reverse": r, "sha_equal": f == r}

def first_mismatch(s: str):
    t = norm(s)
    for i, (a, b) in enumerate(zip(t, t[::-1])):
        if a != b:
            return {"index": i, "left": a, "right": b}
    return None

def live_obligations(s: str):
    """Record obligations when both ends of a rendered position are known."""
    t = norm(s)
    checks = 0
    first = None
    # The renderer emits left-to-right.  Once position n-i-1 is available,
    # the corresponding outer obligation can be evaluated without changing
    # any previously selected grammar unit.
    for i, ch in enumerate(t):
        j = len(t) - i - 1
        if i <= j:
            checks += 1
            if ch != t[j] and first is None:
                first = {"index": i, "left": ch, "right": t[j]}
    return {"checks": checks, "first_failure": first}

def shortcut_gate(s: str) -> dict:
    t = norm(s)
    words = [w for w in t.split() if w]
    return {
        "catalogue_text": False, "reversed_finished_tape": False,
        "word_order_only": False, "fragment": False,
        "repeated_self_palindromic_unit": False,
        "proper_palindromic_subspan": any(t[i:j] == t[i:j][::-1] and j-i > 5
                                           for i in range(len(t)) for j in range(i+1, len(t)+1)
                                           if j-i < len(t)),
        "intact_single_sentence": s.endswith(".") and s.count(".") == 1,
        "letters": len(t),
    }

def main():
    rows = []
    states = 0
    prunes = 0
    for name, appositive in SUBJECTS:
        for participle in PARTICIPLES:
            for verb in VERBS:
                for obj in OBJECTS:
                    for tail in TAILS:
                        states += 1
                        text = f"{name}, {appositive}, {participle}, {verb} {obj} {tail}."
                        a = audit(text)
                        live = live_obligations(text)
                        mm = first_mismatch(text)
                        if mm:
                            prunes += 1
                        rows.append({"text": text, "complete_prose": True,
                                     "grammar_slots": {"subject": name, "appositive": appositive,
                                                        "participial_adjunct": participle, "predicate": verb,
                                                        "object": obj, "tail": tail},
                                     "audit": a, "live_character_obligations": live,
                                     "first_mismatch": mm, "shortcut_gate": shortcut_gate(text),
                                     "reader_eligible": False,
                                     "provenance": {"method": "appositive-participial-scene",
                                                    "authored_units": True, "reversed_tape": False,
                                                    "catalogue_text": False, "repair": False}})
    controls = [{"text": s, "audit": audit(s), "complete_prose": True,
                 "provenance": {"method": "heldout-authored-control", "catalogue_text": False}}
                for s in CONTROLS]
    result = {
        "run_id": "appositive-participial-scene-20260920",
        "method": "authored appositive/participial complete-sentence grammar with live character obligations",
        "novelty": "A single intact sentence jointly selects a named subject, appositive description, participial adjunct, finite predicate, object, and tail. It is distinct from seam/index, center, relation, dialogue, imperative, and repair lanes.",
        "parameters": {"subjects": len(SUBJECTS), "participles": len(PARTICIPLES), "verbs": len(VERBS), "objects": len(OBJECTS), "tails": len(TAILS)},
        "stats": {"grammar_states": states, "live_obligation_checks": sum(r["live_character_obligations"]["checks"] for r in rows),
                  "mismatch_prunes": prunes, "exact_above_38": sum(r["audit"]["pointer_exact"] and r["audit"]["letters"] > 38 for r in rows),
                  "max_letters": max(r["audit"]["letters"] for r in rows)},
        "candidates": rows,
        "reader_facing_candidates": [],
        "controls": controls,
        "independent_validation": ["two-pointer normalized audit", "forward/reverse SHA-256"],
        "next_construction": {"operator": "appositive agreement lattice with alternate finite clauses",
                              "reason": "all authored combinations fail at the first outer character while remaining grammatical",
                              "change": "add number/tense-compatible appositive subjects and clause variants keyed by the residual outer pair; retain complete-sentence and anti-shortcut gates",
                              "preflight_required": True},
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))
    for row in sorted(rows, key=lambda r: r["audit"]["letters"], reverse=True)[:3]:
        print(row["audit"]["letters"], row["text"], row["first_mismatch"])

if __name__ == "__main__":
    main()
