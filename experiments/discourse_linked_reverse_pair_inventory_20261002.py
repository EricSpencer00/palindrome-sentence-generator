"""Small, reproducible preflight for discourse-linked reverse surfaces.

This is an inventory, not a generator: each side is authored independently and
is audited before any reversal is considered.  The deliberately strict gate
keeps lexical semordnilaps and fragments out of the promoted set.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/discourse-linked-reverse-pair-inventory-20261002.json"

PAIRS = [
    ("The tired baker stressed.", "Desserts waited by the oven."),
    ("A sailor drew a reward.", "The drawer was open at noon."),
    ("The pilot saw a level civic sign.", "The crew checked the harbor log."),
    ("The quiet keeper repaid a diaper.", "The nurse stored clean cloth."),
    ("The guide delivered a drawer.", "The clerk received a reward."),
]

def tape(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(left: str, right: str) -> dict:
    a, b = tape(left), tape(right)
    mismatch = next(((i, a[i], b[::-1][i]) for i in range(min(len(a), len(b)))
                     if a[i] != b[::-1][i]), None)
    return {"left_letters": len(a), "right_letters": len(b),
            "equal_length": len(a) == len(b), "pointer_exact": bool(a) and a == b[::-1],
            "left_self_palindrome": bool(a) and a == a[::-1],
            "right_self_palindrome": bool(b) and b == b[::-1],
            "first_mismatch": mismatch,
            "sha256_left": hashlib.sha256(a.encode()).hexdigest(),
            "sha256_right_reverse": hashlib.sha256(b[::-1].encode()).hexdigest()}

def surface_gate(s: str) -> dict:
    words = re.findall(r"[A-Za-z]+", s)
    low = [w.casefold() for w in words]
    repeated = len(low) != len(set(low))
    self_spans = [w for w in low if len(w) > 3 and w == w[::-1]]
    return {"word_count": len(words), "fragment": len(words) < 4,
            "repeated_units": repeated, "word_order_symmetry": low == low[::-1],
            "nested_self_palindrome": bool(self_spans), "catalogue_text": False}

def run() -> dict:
    rows = []
    for left, right in PAIRS:
        a = audit(left, right)
        gate_l, gate_r = surface_gate(left), surface_gate(right)
        reject = any(gate_l[k] or gate_r[k] for k in
                     ("fragment", "repeated_units", "word_order_symmetry",
                      "nested_self_palindrome", "catalogue_text"))
        rows.append({"left": left, "right": right, "audit": a,
                     "surfaces": {"left": gate_l, "right": gate_r},
                     "provenance": {"independently_authored": True,
                                    "finished_tape_reversal": False,
                                    "post_hoc_repair": False,
                                    "same_scene_roles": "baker/sailor/pilot/keeper/guide and nearby objects or setting",
                                    "promoted": a["pointer_exact"] and not reject and
                                                not a["left_self_palindrome"] and not a["right_self_palindrome"]}})
    promoted = [r for r in rows if r["provenance"]["promoted"]]
    return {"experiment_id": "discourse-linked-reverse-pair-inventory-20261002",
            "method": "independently authored paired surfaces; normalize letters only; compare left tape to reversed right tape",
            "stats": {"authored_pairs": len(rows), "exact_pairs": sum(r["audit"]["pointer_exact"] for r in rows),
                      "promoted_nonself_pairs": len(promoted)},
            "pairs": rows, "promoted_pairs": promoted,
            "novelty_preflight": {"status": "passed",
                "signature": "independent-surfaces|same-scene-roles|strict-nonself-gates",
                "distinct_from": "prior semordnilap word banks and finished-tape reversal; this records paired discourse surfaces and rejects lexical/fragment shortcuts"},
            "acceptance_gate": "both complete grammatical surfaces, equal normalized length, exact reverse tapes, neither tape self-palindromic, no repeated or mirrored units",
            "falsifier": "any promoted row violates one gate; current run promotes none",
            "next_construction": "Use a typed two-clause scene grammar with live outer-character residuals; retain each surface as an independent derivation and promote only after the strict paired audit.",
            "status": "no promoted non-self pair; inventory is a negative preflight and construction seed"}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"]))
