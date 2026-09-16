#!/usr/bin/env python3
"""Preflight and bounded test for an append-preserving clause algebra.

The proposed invariant is deliberately stronger than lane 9: after a closed
state, appending one fresh semantic macro must leave the character tape closed
without nesting a mirrored span or repeating a unit.  The algebra exposes its
necessary condition (the appended macro must itself be a palindrome), so the
run is useful even when the condition fails.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/arbitrary-clause-macro-algebra-20260916.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
ID = "arbitrary-clause-macro-algebra-20260916"
SIGNATURE = "finite-fresh-semantic-clause-macros|append-preserving-character-balance|no-mirrored-span|no-unit-repeat"

MACROS = (
    ("observe", "Mara observes the harbor lantern"),
    ("repair", "Jon repairs the western gate"),
    ("record", "Iris records the morning tide"),
    ("carry", "Noah carries a copper compass"),
)

def tape(s: str) -> str: return re.sub(r"[^a-z]", "", s.lower())

def audit(s: str) -> dict:
    t = tape(s); mismatches = []
    for i in range(len(t)//2):
        if t[i] != t[-1-i]: mismatches.append(i)
    f = hashlib.sha256(t.encode()).hexdigest(); r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "two_pointer_exact": not mismatches and bool(t),
            "first_mismatch_offset": mismatches[0] if mismatches else None,
            "sha256_forward": f, "sha256_reverse": r, "sha256_exact": f == r}

def run() -> dict:
    # A one-letter centre is the only non-empty closed seed; no macro is used
    # as a hidden palindrome or copied into a second arm.
    states = []
    current = "a"
    for depth, (name, clause) in enumerate(MACROS, 1):
        before = audit(current)
        current = current + " " + clause + "."
        after = audit(current)
        states.append({"depth": depth, "macro": name, "rendered": current,
                       "before": before, "after": after,
                       "macro_tape": tape(clause),
                       "macro_self_palindrome": tape(clause) == tape(clause)[::-1],
                       "invariant_preserved": before["two_pointer_exact"] and after["two_pointer_exact"],
                       "semantic_macro_fresh": True, "mirrored_span_nested": False,
                       "repeated_unit": False})
    return states

def main() -> None:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [e["id"] for e in entries if e.get("id") != ID and e.get("signature") == SIGNATURE]
    preflight = {"entries_inspected": len(entries), "exact_signature_collisions": collisions,
                 "passed": not collisions, "contrast": {"lane_9_flat_grammar": "ordinary increments but no append-preserving invariant",
                 "grammar_pair_composition": "two arms with independently authored right-arm repair/order",
                 "this_algebra": "single-arm finite macro append with explicit closure invariant"}}
    if collisions: raise RuntimeError(collisions)
    states = run()
    result = {"experiment_id": ID, "signature": SIGNATURE, "novelty_preflight": preflight,
              "states": states, "exact_count": sum(s["after"]["two_pointer_exact"] and s["after"]["sha256_exact"] for s in states),
              "invariant": "closed(P + M) iff reverse(M) equals the suffix obligation induced by P; for arbitrary P-independent append this requires M itself be palindromic",
              "conclusion": "failed_scalable_nonpalindromic_append_invariant",
              "repair_operator": "replace append with a typed boundary solver that chooses a fresh macro whose tape exactly satisfies the live suffix obligation; if none exists, stop and report debt (never repeat or mirror a macro)",
              "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "fresh_hand_authored_macros": True, "catalogue_imported": False, "pre_existing_palindrome_wrapped": False, "mirrored_spans": False, "repeated_units": False},
              "reader_eligible": False}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"exact_count": result["exact_count"], "lengths": [s["after"]["letters"] for s in states], "status": result["conclusion"]}))

if __name__ == "__main__": main()
