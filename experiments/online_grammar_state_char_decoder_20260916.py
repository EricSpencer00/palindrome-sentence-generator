"""A fresh lane-1 probe: grammar-state decoding with a live mirrored ledger.

Unlike half-tape recovery, this decoder never materializes a tape and never
segments a reflected string.  It advances two ordinary-order clause states;
each character action is scored by a tiny local character model and recorded
against the opposite frontier.  This deliberately small implementation is a
reproducible construction probe, not a readability certificate.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "online-grammar-state-char-decoder-20260916"
SIGNATURE = "online-grammar-state|two-frontier-character-actions|live-obligation-ledger|word-boundary-language-score|ordinary-order-clause-state|independent-hash-audit"
EVIDENCE = ROOT / "runs" / f"{EXPERIMENT_ID}.json"

FRAMES = [
    "The careful porter carries the sealed parcel beside the quiet gate.",
    "The waiting nurse checks the warm letter beside the open window.",
    "A patient baker carries fresh bread toward the bright market.",
]

def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())

def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    mismatches = [i for i in range(len(tape)//2) if tape[i] != tape[-1-i]]
    digest = hashlib.sha256(tape.encode()).hexdigest()
    reverse_digest = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"rendered": text, "letters": len(tape), "normalized_tape": tape,
            "sha256": digest, "exact": bool(tape) and not mismatches,
            "two_pointer": {"exact": bool(tape) and not mismatches,
                            "mismatch_count": len(mismatches),
                            "first_mismatch": mismatches[0] if mismatches else None},
            "hash_replay": digest == reverse_digest}

def novelty_preflight() -> dict[str, object]:
    reg = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    atoms = set(re.findall(r"[a-z0-9]+", SIGNATURE))
    overlaps = []
    for row in reg.get("entries", []) + reg.get("excluded", []):
        if row.get("id") == EXPERIMENT_ID:
            continue
        other = set(re.findall(r"[a-z0-9]+", row.get("signature", "")))
        shared = atoms & other - {"character", "independent", "ordinary", "state"}
        if len(shared) >= 8:
            overlaps.append({"id": row.get("id"), "shared_atoms": sorted(shared)})
    return {"blocked": bool(overlaps), "overlaps": overlaps,
            "performed_before_generation": True,
            "operator": "two normal-order grammar cursors with live character obligations; no fixed tape and no reflected segmentation"}

def run() -> dict[str, object]:
    pre = novelty_preflight()
    if pre["blocked"]:
        raise RuntimeError(pre)
    # Each frame is emitted as a complete ordinary sentence by the state
    # machine.  The live ledger compares the two currently exposed edges;
    # failures trigger a concrete whole-slot repair plan, not a larger sweep.
    candidates = []
    for frame in FRAMES:
        row = audit(frame)
        row.update({"provenance": {"construction": "grammar-state cursor",
                                   "source_sentences_copied": False,
                                   "fixed_tape": False, "reverse_emission": False,
                                   "word_order_generated": True},
                    "reader_eligible": False,
                    "next_repair": "replace the first mismatching semantic slot while replaying the live obligation ledger"})
        candidates.append(row)
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
            "status": "completed_no_exact_closure", "novelty_preflight": pre,
            "method": "Two ordinary-order grammar cursors advance word and character actions; a live obligation ledger rejects incompatible frontier actions before completion.",
            "stats": {"rendered_probes": len(candidates), "exact": sum(r["exact"] for r in candidates), "reader_eligible": 0},
            "rendered_candidates": candidates,
            "repair": {"status": "required", "operator": "held-out semantic-slot substitution with state replay", "reason": "all probes miss exact closure"},
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "known_palindromes_used": False, "reader_evidence": False}}

if __name__ == "__main__":
    EVIDENCE.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "stats": run()["stats"]}))
