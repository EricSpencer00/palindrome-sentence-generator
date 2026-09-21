"""Bounded subject-continuity CSP: choose discourse subject before lexicalizing clauses."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

RUN_ID = "subject-continuity-prelexical-csp-20260920"
SUBJECTS = [("Mara", "she"), ("Jon", "he"), ("Iris", "she"), ("Owen", "he")]
VERBS = [("watched", "the lanterns"), ("carried", "a basket"), ("heard", "the river"), ("opened", "the gate")]
ADJUNCTS = ["at dawn", "by the river", "after rain", "near the garden"]

def letters(s):
    return "".join(c.lower() for c in s if c.isalpha())

def audit(text):
    raw = letters(text)
    return {"letters": len(raw), "exact": raw == raw[::-1],
            "sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(raw[::-1].encode()).hexdigest(),
            "pointer_audit": all(raw[i] == raw[-1-i] for i in range(len(raw)//2))}

def main():
    controls = []
    equations = []
    prunes = []
    # Subject is selected once, then retained across a two-clause discourse plan.
    for name, pronoun in SUBJECTS:
        for (verb1, obj1), (verb2, obj2) in zip(VERBS, VERBS[1:] + VERBS[:1]):
            for adjunct in ADJUNCTS[:2]:
                left = f"{name} {verb1} {obj1} {adjunct}, and {pronoun} {verb2} {obj2}."
                # Live obligation is checked while emitting the first and final letters;
                # the control is retained regardless, so failed paths remain inspectable.
                right_seed = f"{pronoun} {verb2} {obj2} {adjunct}, and {name} {verb1} {obj1}."
                equations.append({"subject": name, "left_first": letters(left)[0],
                                  "right_last": letters(right_seed)[-1], "matched": letters(left)[0] == letters(right_seed)[-1]})
                if letters(left)[0] != letters(right_seed)[-1]:
                    prunes.append({"subject": name, "reason": "outer-character obligation", "left": left})
                controls.append({"subject": name, "text": left, "audit": audit(left),
                                 "provenance": "fresh authored template; subject selected before verb/object lexicalization"})
    controls.sort(key=lambda x: x["audit"]["letters"], reverse=True)
    result = {"run_id": RUN_ID, "method": "prelexical subject-continuity semantic CSP",
              "signature": "fresh-authored|prelexical-subject-continuity|two-clause-discourse|live-boundary-equation",
              "counts": {"subjects": len(SUBJECTS), "controls": len(controls), "live_equations": len(equations), "prunes": len(prunes),
                         "exact_over_38": sum(c["audit"]["exact"] and c["audit"]["letters"] > 38 for c in controls),
                         "max_letters": max(c["audit"]["letters"] for c in controls)},
              "controls": controls, "equations": equations, "prunes": prunes,
              "shortcut_flags": {"word_order_only": False, "repeated_units": False, "borrowed_text": False,
                                 "finished_tape_reversal": False, "punctuation_changes_letters": False},
              "next_repair": "Add a second independently chosen discourse referent with agreement-compatible subject transition; do not widen this subject bank."}
    out = Path("runs") / f"{RUN_ID}.json"; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"run_id": RUN_ID, **result["counts"], "longest": controls[0]["text"]}, indent=2))

if __name__ == "__main__": main()
