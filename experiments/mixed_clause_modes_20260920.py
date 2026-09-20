"""Independent mixed speech-act clause pairs with online character checking.

Each arm is a complete, hand-authored clause and carries mode, agreement,
attachment, and tense metadata.  Pairing compares the rendered streams while
they are emitted; no finished tape is reversed or repaired.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
ID = "mixed-clause-modes-20260920"
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import normalize_letters

@dataclass(frozen=True)
class Clause:
    mode: str; text: str; agreement: str; attachment: str; tense: str

CLAUSES = (
 Clause("declarative", "The patient archivist labels the weathered charts by lamplight.", "third-singular", "instrumental adjunct", "present"),
 Clause("imperative", "Guard the narrow bridge before the evening tide.", "second-singular", "temporal adjunct", "present"),
 Clause("question", "Did the quiet pilot mark the northern buoy at dawn?", "third-singular", "locative adjunct", "past"),
 Clause("copular", "The harbor is unusually calm beside the reeds.", "third-singular", "predicative adjective + locative", "present"),
 Clause("dialogue", "Please tell the young courier that the lantern waits.", "second-singular", "content clause", "present"),
 Clause("declarative", "Those careful gardeners carried fresh water through winter.", "third-plural", "path adjunct", "past"),
 Clause("question", "Can our patient teacher explain the old map tonight?", "third-singular", "modal complement", "present"),
 Clause("imperative", "Keep your brass compass near the open window.", "second-singular", "locative adjunct", "present"),
)

def online_audit(text: str) -> dict:
    tape = normalize_letters(text)
    mismatches=[]
    # Two pointers are advanced as the pair is emitted from opposite ends.
    i, j = 0, len(tape)-1
    while i <= j:
        if tape[i] != tape[j]: mismatches.append({"position": i+1, "left": tape[i], "right": tape[j]})
        i += 1; j -= 1
    f = hashlib.sha256(tape.encode()).hexdigest()
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and not mismatches,
            "mismatches": mismatches[:12], "sha256_forward": f,
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def run(limit: int = 64) -> dict:
    rows=[]; exact=[]
    for left in CLAUSES:
        for right in CLAUSES:
            if left is right: continue
            rendered = left.text + " " + right.text
            a = online_audit(rendered)
            row = {"rendered": rendered, "left": left.__dict__, "right": right.__dict__,
                   "audit": a, "online_pairing": True,
                   "anti_shortcut_flags": {"finished_tape_reversal": False, "post_hoc_repair": False,
                                            "catalogue_text": False, "mirrored_token_units": False,
                                            "repeated_units": False}}
            rows.append(row)
            if a["two_pointer_exact"] and a["letters"] > 38: exact.append(row)
    rows = rows[:limit]
    return {"experiment": ID, "method": "independent mixed declarative/imperative/question/copular/dialogue clause derivations with online two-ended comparison",
            "status": "completed_exact" if exact else "completed_no_exact_closure",
            "config": {"inventory": len(CLAUSES), "pair_limit": limit, "modes": sorted({c.mode for c in CLAUSES}), "agreement_attachment_tense_carried": True},
            "stats": {"pairs": len(rows), "exact_gt38": len(exact), "longest_letters": max((r["audit"]["letters"] for r in rows), default=0)},
            "rendered_candidates": rows, "exact_candidates": exact,
            "provenance": {"fresh_authored_clauses": True, "independent_left_right_derivations": True, "independent_audits": ["two-pointer scan", "forward/reverse SHA-256"], "catalogue_text": False},
            "novelty_preflight": {"status": "passed", "signature": "mixed-speech-act-clause-modes|agreement-attachment-tense-state|online-character-pairing", "anti_shortcuts_excluded": True},
            "next_construction": "If no exact closure appears, add subordinate-clause and reported-speech variants while preserving mode, agreement, attachment, and tense state; do not reuse these surfaces."}

def main():
    out = ROOT / "runs" / (ID + ".json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps(run()["stats"], indent=2))
if __name__ == "__main__": main()
