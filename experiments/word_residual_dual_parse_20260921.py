"""Bounded productive word-residual dual-parse experiment."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
from llm_palindrome.dual_parse import word_residual_search, letter_tape
from llm_palindrome.admission import mechanical_admission_checks

ID = "word-residual-dual-parse-20260921"
LEFT = (("A:determiner", ("an",)), ("A:agent", ("aide",)),
        ("B:verb", ("rips",)), ("B:quantity", ("nine",)),
        ("B:object", ("memos",)))
RIGHT = (("B-prime:response", ("some",)), ("B-prime:agent", ("men",)),
         ("B-prime:verb", ("inspire",)), ("A-prime:patient", ("Diana",)))

def audit(text: str) -> dict:
    tape = letter_tape(text)
    mismatch = [(i, len(tape)-1-i) for i in range(len(tape)//2)
                if tape[i] != tape[-1-i]]
    return {"letters": len(tape), "pointer_exact": not mismatch,
            "mismatches": mismatch, "normalized_tape": tape,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def run() -> dict:
    search = word_residual_search(LEFT, RIGHT, max_states=2_000, max_results=50)
    rows = []
    for result in search["results"]:
        text = result["rendered"]
        checks = mechanical_admission_checks(text, min_letters=39, max_letters=100)
        rows.append({**result, "audit": audit(text), "mechanical_admission": checks,
                     "provenance": {"word_residual_state": True,
                                    "joint_role_tracking": True,
                                    "finished_tape_reversal": False,
                                    "completed_prose_enumeration": False,
                                    "reversible_units": False,
                                    "known_lane_duplicate": False}})
    exact = [r for r in rows if all(r["mechanical_admission"].values())]
    return {"experiment_id": ID,
            "method": "bounded online word-residual intersection of independent POS/role plans",
            "stats": {"states": search["states"], "transitions": search["transitions"],
                      "rendered": len(rows), "exact_gt38": len(exact),
                      "max_letters": max((r["audit"]["letters"] for r in rows), default=0)},
            "exact_candidates": exact, "controls": rows,
            "novelty_preflight": {"status": "recovery_control_only", "signature": ID,
                                  "reason": "the fixed lexical inventory verifies word-residual orientation; productive alternatives belong in a separate search"},
            "provenance": {"audits": ["independent two-pointer comparison", "forward/reverse SHA-256"],
                           "central_mechanical_admission": True,
                           "reader_gate": "exact >38 only", "hard_exclusions": ["nested palindromes", "repeated units", "word-order symmetry", "catalogue text"]},
            "status": "fresh exact >38 requires reading" if exact else "38-letter recovery control only; no exact clean >38 closure"}

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "runs" / f"{ID}.json"
    out.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps(run(), indent=2))
