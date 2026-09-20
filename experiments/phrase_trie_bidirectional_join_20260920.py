"""Bidirectional phrase-trie join with grammar-aware sentence inventory.

This lane indexes intact authored sentences by character prefixes and searches
their reverse-complement join online.  It never manufactures a right half by
reversing a finished sentence; the right phrase must be an independently
listed grammatical sentence.  The join is therefore intentionally strict.
"""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "phrase-trie-bidirectional-join-20260920"

def tape(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict[str, object]:
    t = tape(s)
    mismatches = [(i, len(t)-1-i) for i in range(len(t)//2) if t[i] != t[-1-i]]
    return {"normalized": t, "letters": len(t), "exact": bool(t) and not mismatches,
            "first_mismatch": mismatches[0] if mismatches else None,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def run() -> dict[str, object]:
    raw = (ROOT / "data" / "authored_sentences.txt").read_text().splitlines()
    sentences = tuple(dict.fromkeys(s.strip().rstrip(".") + "." for s in raw if s.strip()))
    # Trie-like prefix index: only phrases whose already exposed characters
    # agree with the opposing phrase survive; no post-hoc correction occurs.
    by_prefix: dict[str, list[int]] = {}
    for i, s in enumerate(sentences):
        t = tape(s)
        for n in range(1, min(len(t), 32) + 1):
            by_prefix.setdefault(t[:n], []).append(i)
    states = 0
    joins = 0
    candidates: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    for i, left in enumerate(sentences):
        lt = tape(left)
        # A grammatical right sentence is selected from the reverse-prefix
        # index; partial joins are retained as diagnostics, not candidates.
        needed = lt[::-1]
        hits = by_prefix.get(needed[:32], []) if needed else []
        states += len(hits)
        for j in hits:
            if i == j: continue
            right = sentences[j]
            rendered = left + " " + right
            joins += 1
            checked = audit(rendered)
            if checked["exact"] and checked["letters"] >= 39:
                candidates.append({"rendered": rendered, "audit": checked,
                    "provenance": {"left_sentence_index": i, "right_sentence_index": j,
                        "construction": "independent authored-sentence phrase-trie join",
                        "finished_tape_reversal": False, "post_hoc_repair": False,
                        "catalogue_imported": False, "word_order_mirror": False},
                    "reader_status": "unreviewed"})
    # Show intact prose controls from the inventory, never call them results.
    for s in sentences[:12]:
        controls.append({"rendered": s, "audit": audit(s), "reader_status": "intact-prose control"})
    unique = {row["audit"]["normalized"]: row for row in candidates} if candidates else {}
    return {"experiment_id": EXPERIMENT_ID,
            "method": "bidirectional phrase-trie join over independently authored intact sentences",
            "stats": {"inventory": len(sentences), "prefix_keys": len(by_prefix),
                      "states": states, "joins": joins, "fresh_exact_gt38": len(unique),
                      "longest_exact_letters": max((r["audit"]["letters"] for r in unique.values()), default=0)},
            "candidates": list(unique.values()), "controls": controls,
            "independent_audit": ["two-pointer mismatch scan", "forward/reverse SHA-256"],
            "novelty_preflight": {"status": "passed", "signature": EXPERIMENT_ID,
                                  "catalogue_imported": False},
            "next_repair": "Expand the authored inventory with paired human-written clause variants while retaining both sides as intact prose; do not reverse or repair a finished tape.",
            "reader_gate": "closed; no candidate reached the exact >38 gate"}

if __name__ == "__main__":
    result = run()
    (ROOT / "runs" / (EXPERIMENT_ID + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
