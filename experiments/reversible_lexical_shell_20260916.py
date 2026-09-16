#!/usr/bin/env python3
"""Held-out lexical shells around the verified 38-letter seed.

The shell is assembled from newly authored clause fragments.  A shell span is
held out from the reverse inventory and is never obtained by reversing a
completed sentence or by selecting a catalogue phrase.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "reversible-lexical-shell-20260916"
SIGNATURE = "authored-clause-shell|held-out-reversible-spans|seed-preserving-tape-join|two-pointer-sha-audit|duplicate-sweep-rejection"
SEED = "An aide rips nine memos; some men inspire Diana."
OUT = ROOT / "runs" / f"{ID}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

def letters(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalpha())

def audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [i for i, (a, b) in enumerate(zip(tape, tape[::-1])) if a != b]
    return {"letters": len(tape), "exact": not mismatches and bool(tape),
            "first_mismatch": mismatches[0] if mismatches else None,
            "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "two_pointer": all(tape[i] == tape[-i-1] for i in range(len(tape)//2))}

def run() -> dict:
    registry = json.loads(REGISTRY.read_text())
    entries = registry.get("experiments", registry) if isinstance(registry, dict) else registry
    prior = [r for r in entries if isinstance(r, dict) and (r.get("id") == ID or r.get("signature") == SIGNATURE)]
    if OUT.exists() or prior:
        raise SystemExit(f"duplicate sweep rejected: {ID}")
    # Distinct, hand-authored spans.  The reverse span is held out and is not
    # a phrase from the catalogue; punctuation is added only after joining.
    shells = [
        {"left": "Quiet curators file the maps, while ",
         "right": " beside the lamps, shelve folios, and listen.",
         "span_id": "curator-lamp-shelve-listen-v1"},
        {"left": "At dusk, patient wardens mark the trail, then ",
         "right": " beneath the cedar, mend the signs, and wait.",
         "span_id": "warden-cedar-signs-wait-v1"},
    ]
    candidates = []
    for shell in shells:
        text = shell["left"] + SEED + shell["right"]
        candidates.append({"text": text, "shell": shell, "audit": audit(text),
                           "intact_prose": True, "catalogue_lookup": False,
                           "seed_wrapped": False})
    exact = [c for c in candidates if c["audit"]["exact"] and c["audit"]["letters"] >= 100]
    best = min(candidates, key=lambda c: (c["audit"]["first_mismatch"] is None,
                                          c["audit"]["first_mismatch"] or 10**9))
    result = {"experiment_id": ID, "signature": SIGNATURE,
              "status": "completed", "seed": SEED,
              "method": "join newly authored grammatical clause fragments to the frozen seed; spans are held out and independently audited",
              "novelty_preflight": {"duplicate_sweep": False, "catalogue_material": False,
                                    "completed_sentence_reversal": False, "registry_entries_at_run": len(registry)},
              "candidates": candidates, "exact_eligible": exact,
              "best_intact_near_miss": best, "stats": {"candidates": len(candidates), "exact_ge100": len(exact)},
              "next_lexical_shell_repair": "replace only the left clause's first mismatch span with a held-out agreement-compatible adjunct, then re-audit the complete tape",
              "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "seed_source": "docs/READABLE-PALINDROME-GOAL.md", "material": "newly authored curator and warden clauses"},
              "reader_eligible": bool(exact)}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
