"""Small exhaustive certificate: distinguish endpoint starvation from search failure.

No model, catalogue, reversal construction, or text repair is used. This is
diagnostic evidence about two frozen 54-derivation grammars, not discovery.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.typed_boundary_valency_20260921 import SUBJECTS, VERBS, OBJECTS, LOCATIVES, TAILS, norm


def audit(text):
    tape = norm(text)
    mismatch = next((i for i in range(len(tape) // 2)
                     if tape[i] != tape[-1-i]), None)
    return {"letters": len(tape), "pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch_pair": mismatch,
            "forward_sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest()}


def paired_prefix(left, right):
    """Compare independent terminal sequences in their actual surface order.

    Return matched characters and the next opposing characters, carrying
    character positions across unequal terminal boundaries. Terminal reversal
    is traversal only; it never authors any output text.
    """
    forward = ((c, wi, ci) for wi, word in enumerate(left)
               for ci, c in enumerate(norm(word)))
    backward = ((norm(right[wi])[ci], wi, ci)
                for wi in range(len(right)-1, -1, -1)
                for ci in range(len(norm(right[wi]))-1, -1, -1))
    matched = 0
    for a, b in itertools.zip_longest(forward, backward):
        if a is None or b is None:
            return {"matched": matched, "mismatch": None, "exhausted_side": "left" if a is None else "right"}
        if a[0] != b[0]:
            return {"matched": matched, "mismatch": {"left": a, "right": b}, "exhausted_side": None}
        matched += 1
    return {"matched": matched, "mismatch": None, "exhausted_side": "both"}


def certificate(tails):
    # Exterior slots are independent of verb/complement choices. Reject an
    # incompatible exterior before instantiating either interior slot.
    support = []
    for (subject, number), tail in itertools.product(SUBJECTS, tails):
        check = paired_prefix([subject], [tail])
        support.append({"subject": subject, "tail": tail, **check})
    rows = []
    for subject, number in SUBJECTS:
        for verb, valency in VERBS[number]:
            for complement, tail in itertools.product(OBJECTS if valency == "transitive" else LOCATIVES, tails):
                text = f"{subject} {verb} {complement} {tail}."
                rows.append({"text": text, "audit": audit(text)})
    counts = {str(depth): sum(row["audit"]["first_mismatch_pair"] is None or
                              row["audit"]["first_mismatch_pair"] >= depth for row in rows)
              for depth in range(4)}
    best = max(rows, key=lambda row: row["audit"]["first_mismatch_pair"] or 0)
    return {"domain_derivations": len(rows), "exterior_pairs": support,
            "exterior_pairs_surviving": sum(x["mismatch"] is None for x in support),
            "derivations_surviving_paired_prefix": counts,
            "exact": sum(row["audit"]["pointer_exact"] for row in rows),
            "diagnostic_prose": best, "reader_candidates": []}


def run():
    return {"original_domain": certificate(TAILS),
            "first_character_supported_domain": certificate(["at sunset", "beneath the stars", "in the plaza"]),
            "limits": "Exhaustive only for these two finite 54-derivation grammars; no novel palindrome claimed.",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
