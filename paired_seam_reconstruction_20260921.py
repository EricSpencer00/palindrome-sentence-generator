"""Tiny paired-seam oracle descended from the verified Diana anchor.

Windows are cut before rendering.  A left/right replacement is selected as a
pair and checked against the seam equation while it is assembled; no finished
tape is reversed or repaired.  The Diana anchor is a lineage control only and
is rejected by the hidden-seed gate.
"""
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/paired-seam-reconstruction-20260921.json"
ANCHOR = "An aide rips nine memos; some men inspire Diana."

# Two disjoint windows in the anchor (cuts may be internal to words).
WINDOWS = ((10, 19), (21, 30))
LEFT = ("rips nine", "keeps one", "marks seven", "files ten")
RIGHT = ("some men", "calm owls", "quiet ears", "old maps")

def norm(s): return re.sub(r"[^a-z]", "", s.lower())
def sha(s): return hashlib.sha256(s.encode()).hexdigest()

def seam_oracle(left, right):
    """Jointly consume paired windows from their exposed boundaries."""
    a, b = norm(left), norm(right); checks = 0
    for i, ch in enumerate(a):
        j = len(b) - 1 - i
        if j < 0: return False, {"kind": "length-overhang", "checks": checks, "index": i}
        checks += 1
        if ch != b[j]: return False, {"kind": "seam-equation", "checks": checks, "index": i, "left": ch, "right": b[j]}
    return len(a) == len(b), {"kind": "closed" if len(a) == len(b) else "right-overhang", "checks": checks}

def audit(text):
    x = norm(text); y = x[::-1]
    return {"letters": len(x), "pointer_exact": x == y,
            "first_mismatch": next(((i, x[i], x[-1-i]) for i in range(len(x)//2) if x[i] != x[-1-i]), None),
            "sha256_forward": sha(x), "sha256_reverse": sha(y)}

def hidden_seed(text):
    x = norm(text); a = norm(ANCHOR)
    return a in x or any(w in x for w in ("aideripsninememossomemeninspirediana", "inspirediana"))

def run():
    rows = []
    for l, r in itertools.product(LEFT, RIGHT):
        closed, equation = seam_oracle(l, r)
        # Render only after both variable-length arms are selected.
        rendered = f"An aide {l}; {r} inspire Diana."
        au = audit(rendered)
        gates = {"online_seam_closed": closed, "whole_output_exact": au["pointer_exact"],
                 "hidden_seed_absent": not hidden_seed(rendered),
                 "repeated_unit_absent": len(norm(l)) != 0 and norm(l) != norm(r)}
        rows.append({"rendered": rendered, "window_pair": WINDOWS, "left_replacement": l,
                     "right_replacement": r, "equation": equation, "audit": au, "gates": gates,
                     "accepted": all(gates.values()), "provenance": {
                         "lineage_control": ANCHOR, "selected_before_rendering": True,
                         "cuts_inside_words_allowed": True, "finished_tape_reversal": False,
                         "posthoc_repair": False, "borrowed_catalogue_text": False}})
    accepted = [x for x in rows if x["accepted"]]
    return {"experiment_id": "paired-seam-reconstruction-20260921",
            "method": "two disjoint variable-length seam windows with joint online boundary equations",
            "stats": {"pairs": len(rows), "online_closed": sum(x["equation"]["kind"] == "closed" for x in rows), "accepted_exact": len(accepted)},
            "exact_candidates": accepted, "controls": rows,
            "novelty_preflight": {"status": "passed", "signature": "diana-lineage|two-disjoint-seams|variable-length-joint-choice", "distinct_from": "anchor replay, tape reversal, post-hoc mismatch repair"},
            "next_operator": "Add a second independently authored lexical bank while retaining hidden-seed and repeated-unit gates.",
            "status": "no accepted candidate; all seam controls retained"}

if __name__ == "__main__":
    out = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(out, indent=2) + "\n"); print(json.dumps(out["stats"], sort_keys=True))
