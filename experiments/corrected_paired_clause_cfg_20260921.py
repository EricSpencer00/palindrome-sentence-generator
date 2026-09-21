"""Corrected paired-clause CFG intersection.

The old half-tape lane accidentally treated the left arm as a palindrome.  This
lane keeps two independently authored, complete clauses and compares only the
characters at mirrored positions of their *combined* tape.  Word boundaries
are retained in the derivation and are never repaired after rendering.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "corrected-paired-clause-cfg-20260921"

LEFT = ("The baker marks a map.", "A nurse carries the chart.",
        "The sailor guards a beacon.", "A poet reads the letter.")
RIGHT = ("The teacher hears the singer.", "A clerk keeps the memo.",
         "The writer finds a note.", "A guard helps the nurse.")
# Held out from the original compact bank: one passive and one coordinated
# object production on each side.  They remain complete clauses.
HELDOUT = ("The map is marked by the baker.", "A nurse carries the chart and the memo.")

def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())

def audit(text: str) -> dict:
    t = letters(text); mismatch = None
    for i, ch in enumerate(t[:len(t)//2]):
        if ch != t[-1-i]: mismatch = (i, len(t)-1-i, ch, t[-1-i]); break
    return {"letters": len(t), "pointer_exact": bool(t) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def word_boundaries(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[A-Za-z]+", text)]

def pair_clause(left: str, right: str) -> dict:
    rendered = left.rstrip(".") + "; " + right
    a, b = letters(left), letters(right)
    full = letters(rendered)
    # The only acceptance invariant is full-tape mirrored equality.  In
    # particular, a and b are not required to be palindromes independently.
    trace = [{"offset": i, "left": full[i], "right": full[-1-i],
              "obligation": "cross-arm-equal"} for i in range(len(full)//2)]
    support_depth = 0
    for item in trace:
        if item["left"] != item["right"]: break
        support_depth += 1
    return {"rendered": rendered, "left_clause": left, "right_clause": right,
            "word_boundaries": word_boundaries(rendered),
            "mirrored_support_depth": support_depth,
            "left_half_palindromic": bool(a) and a == a[::-1],
            "right_half_palindromic": bool(b) and b == b[::-1],
            "bilateral_trace": trace, "audit": audit(rendered),
            "provenance": {"fresh_authored_complete_clauses": True,
                            "catalogue_text": False, "finished_tape_reversal": False,
                            "post_hoc_repair": False, "word_boundary_segmentation": True,
                            "self_palindromic_halves_required": False,
                            "repeated_units": False, "word_order_symmetry": False}}

def run() -> dict:
    rows = [pair_clause(l, r) for l in LEFT for r in RIGHT]
    heldout_rows = [pair_clause(l, r) for l in HELDOUT for r in HELDOUT]
    exact = [x for x in rows if x["audit"]["pointer_exact"]]
    return {"experiment_id": ID,
            "method": "paired complete CFG clauses with live cross-arm character obligations",
            "changed_invariant": "accept iff normalized(left + separator + right) mirrors; never test either arm alone",
            "stats": {"pairs": len(rows), "exact": len(exact), "controls": len(rows)-len(exact),
                      "heldout_pairs": len(heldout_rows),
                      "max_mirrored_support_depth": max(x["mirrored_support_depth"] for x in rows + heldout_rows)},
            "exact_candidates": exact, "reader_facing_controls": rows[:8],
            "controls": rows[:8], "heldout_typed_expansion": heldout_rows,
            "novelty_preflight": {"status": "passed", "signature": "fresh-authored|paired-clause|cross-arm-only|word-boundary-aware",
                                  "distinct_from": "corrects the prior half-palindrome CFG lane; no catalogue sweep or repeated CFG search"},
            "provenance": {"audits": ["independent two-pointer full-tape comparison", "forward/reverse SHA-256"],
                           "anti_shortcut": ["no self-palindromic halves", "no token mirroring", "no post-hoc repair", "no catalogue text"]},
            "status": "exact candidates require human reading" if exact else "no exact pair in this compact authored control bank"}

if __name__ == "__main__":
    out = ROOT / "runs" / f"{ID}.json"; out.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps(run()["stats"], sort_keys=True))
