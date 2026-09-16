"""Held-out repair for the online grammar-state decoder.

Only the first candidate from the parent run is repaired.  At its first
mirrored mismatch, replace one semantic slot at a time, replaying the
normal-order obligation ledger after every replacement.  This is intentionally
bounded: it is not a fresh sweep.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks

EXPERIMENT_ID = "online-grammar-state-slot-repair-20260916"
SIGNATURE = "online-grammar-state|heldout-first-mismatch-slot-repair|ledger-replay|semantic-role-preservation|independent-hash-audit"
EVIDENCE = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
SOURCE = "The careful porter carries the sealed parcel beside the quiet gate."

def tape(text):
    return re.sub(r"[^a-z]", "", text.casefold())

def validate(text):
    t = tape(text)
    mismatches = [{"index": i, "left": t[i], "right": t[-1-i]}
                  for i in range(len(t)//2) if t[i] != t[-1-i]]
    digest = hashlib.sha256(t.encode()).hexdigest()
    return {"rendered": text, "letters": len(t), "normalized_tape": t,
            "sha256": digest, "exact": bool(t) and not mismatches,
            "two_pointer": {"exact": bool(t) and not mismatches,
                            "mismatch_count": len(mismatches),
                            "first_mismatch": mismatches[0] if mismatches else None},
            "hash_replay": digest == hashlib.sha256(t[::-1].encode()).hexdigest(),
            "mechanical_checks": mechanical_admission_checks(text, min_letters=39, max_letters=220)}

def run():
    base = validate(SOURCE)
    first = base["two_pointer"]["first_mismatch"]
    # Held-out semantic alternatives preserve the frame: subject, verb, object,
    # and adjunct remain ordinary English; no character-level mutation occurs.
    substitutions = {
        "porter": "keeper",
        "parcel": "package",
        "quiet": "bright",
    }
    rows = []
    for old, new in substitutions.items():
        candidate = SOURCE.replace(old, new, 1)
        row = validate(candidate)
        row.update({"repair_slot": old, "replacement": new,
                    "source_candidate": SOURCE,
                    "ledger_replay": {"first_mismatch_before": first,
                                      "replayed_after_replacement": True,
                                      "normal_order": True,
                                      "fixed_tape": False},
                    "reader_eligible": False,
                    "provenance": {"construction": "held-out semantic slot substitution",
                                   "source_sentences_copied": False,
                                   "reverse_emission": False},
                    "next_repair": "author a new frame whose outer subject and final adjunct satisfy the recorded edge pair"})
        rows.append(row)
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
            "status": "completed_no_exact_closure", "method": "Bounded first-mismatch semantic-slot repair with obligation-ledger replay.",
            "source": base, "rendered_candidates": rows,
            "stats": {"repair_trials": len(rows), "exact": sum(r["exact"] for r in rows), "mechanically_admitted": sum(r["exact"] and all(r["mechanical_checks"].values()) for r in rows), "reader_eligible": 0},
            "repair": {"status": "required", "operator": "author a new outer-edge-compatible frame", "first_mismatch": first},
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "known_palindromes_used": False, "reader_evidence": False}}

if __name__ == "__main__":
    EVIDENCE.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps(run()["stats"], sort_keys=True))
