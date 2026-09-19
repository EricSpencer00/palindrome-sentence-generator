"""Focused repair for the conditional-omen lane's first terminal debt.

Only the consequence clause is changed; the prior subject/verb/object choice
is held fixed.  This is not a wider Cartesian product.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "luna-half-tape-terminal-repair-20260919.json"
BASE = "When a patient queen foresees a kinder dawn, "
HELD_OUT = ("the waiting realm shall find its courage", "our troubled hearts shall learn their measure")

def norm(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t = norm(s); i, j = 0, len(t)-1; mismatches = []
    while i < j:
        if t[i] != t[j]: mismatches.append({"offset": i, "left": t[i], "right": t[j]})
        i += 1; j -= 1
    f = hashlib.sha256(t.encode()).hexdigest(); r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"normalized_tape": t, "letters": len(t), "two_pointer_exact": bool(t) and not mismatches,
            "mismatch_count": len(mismatches), "first_mismatch": mismatches[0] if mismatches else None,
            "sha256_forward": f, "sha256_reverse": r, "sha256_equal": f == r}

def run():
    rows = []
    for clause in HELD_OUT:
        text = BASE + clause + "."
        a = audit(text)
        rows.append({"rendered": text, "held_out_consequence": clause,
                     "repair_operator": "terminal-compatible consequence substitution",
                     "terminal_target": "e", "audit": a,
                     "provenance": {"base_frame_held_fixed": True, "new_clause_authored": True,
                                     "same_product_not_expanded": True, "posthoc_reversal": False,
                                     "catalogue_text": False},
                     "shortcut_rejection": {"finished_tape_reversal": False,
                                             "word_order_symmetry": False, "repeated_unit": False,
                                             "fragment": False}})
    exact = [r for r in rows if r["audit"]["two_pointer_exact"]]
    return {"experiment_id": "luna-half-tape-terminal-repair-20260919",
            "status": "completed_exact" if exact else "repair_failed_no_exact",
            "candidate_count": len(rows), "exact_count": len(exact),
            "rendered_candidates": rows,
            "failure_and_repair": {"failure": "terminal repair did not close inner debts" if not exact else "exact row requires reader gate",
                                    "next_repair": "Carry the first remaining mirrored debt into a held-out verb-object inflection, not a broader consequence sweep."},
            "independent_audits": ["two-pointer", "forward/reverse SHA-256"],
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "candidates": result["candidate_count"], "exact": result["exact_count"]}))
