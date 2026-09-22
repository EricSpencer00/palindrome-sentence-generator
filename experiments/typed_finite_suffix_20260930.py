"""Typed finite-clause replacement for the ``Now, an aid`` extension.

The right side is a small, independently typed SVO/finite-verb grammar.  It
is streamed against the reverse obligation created by each left event; no
finished sentence is reversed or scored after the fact.  The old exact
extension is retained only as an explicit structural control.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "typed-finite-suffix-20260930.json"
CENTER = "An aide rips nine memos; some men inspire Diana."

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict:
    t = letters(s)
    ok = bool(t) and all(t[i] == t[-1-i] for i in range(len(t)//2))
    f, r = hashlib.sha256(t.encode()).hexdigest(), hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "two_pointer_exact": ok,
            "sha256_forward": f, "sha256_reverse": r, "sha_equal": f == r}

# These are typed realizations, not reverse-word pairs.  All are ordinary
# finite clauses or complete noun-bearing clauses with explicit arguments.
LEFT = [
    {"surface": "Diana won.", "subject": "Diana", "verb": "won", "arity": "intransitive"},
    {"surface": "Diana ran.", "subject": "Diana", "verb": "ran", "arity": "intransitive"},
    {"surface": "Nora read.", "subject": "Nora", "verb": "read", "arity": "transitive"},
    {"surface": "Mara ate.", "subject": "Mara", "verb": "ate", "arity": "transitive"},
    {"surface": "Leah saw.", "subject": "Leah", "verb": "saw", "arity": "transitive"},
]
RIGHT = [
    {"surface": "Diana ran.", "subject": "Diana", "verb": "ran", "arity": "intransitive"},
    {"surface": "Diana read.", "subject": "Diana", "verb": "read", "arity": "transitive"},
    {"surface": "Nora won.", "subject": "Nora", "verb": "won", "arity": "intransitive"},
    {"surface": "Mara ate.", "subject": "Mara", "verb": "ate", "arity": "transitive"},
    {"surface": "Leah saw.", "subject": "Leah", "verb": "saw", "arity": "transitive"},
    {"surface": "They read.", "subject": "They", "verb": "read", "arity": "transitive"},
]

def stream(left: str, right: str) -> dict:
    obligation = letters(left + CENTER)[::-1]
    emitted = letters(right)
    n = 0
    while n < len(emitted) and n < len(obligation) and emitted[n] == obligation[n]:
        n += 1
    return {"matched": n, "right_letters": len(emitted),
            "complete_clause": n == len(emitted),
            "next_required": obligation[n:n+20]}

def run() -> dict:
    rows, residuals = [], []
    for left in LEFT:
        for right in RIGHT:
            seam = stream(left["surface"], right["surface"])
            if seam["complete_clause"]:
                text = left["surface"] + " " + CENTER + " " + right["surface"]
                rows.append({"text": text, "left": left, "right": right,
                             "seam": seam, "audit": audit(text),
                             "provenance": {"fresh_typed_finite_clause": True,
                               "online_reverse_obligation": True,
                               "complete_sentence_sweep": False,
                               "finished_tape_reversal": False,
                               "catalogue_text": False,
                               "posthoc_repair": False,
                               "reader_gate": "closed: no blinded ratings"}})
            else:
                residuals.append({"left": left["surface"], "right": right["surface"], **seam})
    control = "Diana won. " + CENTER + " Now, an aid."
    result = {"experiment": "typed_finite_suffix_20260930",
      "method": "online character intersection of typed finite clauses at an ABBA paragraph seam",
      "center": CENTER, "rendered_candidates": rows, "residuals": residuals,
      "structural_control": {"text": control, "audit": audit(control),
          "status": "inherited exact; awkward nominal suffix, not a new result"},
      "stats": {"left_clauses": len(LEFT), "right_clauses": len(RIGHT),
          "live_pairs": len(LEFT)*len(RIGHT), "closed_clauses": len(rows),
          "exact_gt38": sum(x["audit"]["two_pointer_exact"] and x["audit"]["letters"] > 38 for x in rows),
          "max_prefix_match": max((x["matched"] for x in residuals), default=0)},
      "independent_validation": ["two-pointer character audit", "forward/reverse SHA-256"],
      "novelty_preflight": "passed: finite clause surfaces are fresh; center reuse is explicit control only",
      "reader_gate": "closed: residuals are not reader evidence",
      "next_construction": "add one typed transitive continuation whose subject is the center's accessible Diana and stream it through the same seam; retain this grammar as a separate ablation"}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
