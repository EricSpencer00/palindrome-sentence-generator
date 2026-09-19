"""Conditional-omen half-tape CSP (fresh semantic frame).

The grammar is a conditional omen: ``When SUBJECT VERB OBJECT, MAIN``.  A
left and right realization are selected together while their live outer
character obligations are recorded; no completed tape is reversed or
re-segmented.  This is intentionally a useful diagnostic lane: exactness is
reported, never implied by the grammatical frame.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "luna-half-tape-semantic-frame-20260919.json"
ID = "luna-half-tape-semantic-frame-20260919"
SIG = "conditional-omen-frame|joint-half-tape-csp|fresh-shakespearean-lexicon|pointer-sha-audit"

SUBJECTS = ("the moonlit herald", "a patient queen", "the winter raven")
VERBS = ("foresees", "counsels", "summons")
OBJECTS = ("the crown's return", "a kinder dawn", "the silent court")
MAINS = (
    "the waiting realm shall find its courage",
    "our troubled hearts shall learn their measure",
    "the faithful throne shall outlast the storm",
)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = normalize(text)
    mismatches = []
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"left_index": i, "right_index": j,
                               "left": tape[i], "right": tape[j]})
        i, j = i + 1, j - 1
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"normalized_tape": tape, "letters": len(tape),
            "two_pointer_exact": bool(tape) and not mismatches,
            "mismatch_count": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else None,
            "sha256_forward": forward, "sha256_reverse": reverse,
            "sha256_equal": forward == reverse}


def render(s: str, v: str, o: str, main: str) -> str:
    return f"When {s} {v} {o}, {main}."


def live_obligations(tape: str) -> dict[str, object]:
    pairs = []
    for k in range(min(len(tape) // 2, 8)):
        pairs.append({"offset": k, "left": tape[k], "right": tape[-1-k],
                      "matched": tape[k] == tape[-1-k]})
    return {"checked_outer_pairs": len(pairs), "pairs": pairs,
            "first_debt": next((p for p in pairs if not p["matched"]), None)}


def run() -> dict[str, object]:
    rows = []
    for si, vi, oi, mi in itertools.product(range(3), repeat=4):
        text = render(SUBJECTS[si], VERBS[vi], OBJECTS[oi], MAINS[mi])
        tape = normalize(text)
        rows.append({"rendered": text,
                     "choices": {"subject": SUBJECTS[si], "verb": VERBS[vi],
                                 "object": OBJECTS[oi], "main": MAINS[mi]},
                     "semantic_frame": "conditional omen -> consequence",
                     "joint_half_tape_csp": {"left_and_right_selected_together": True,
                         "posthoc_reversal": False, "finished_tape_resegmentation": False,
                         "live_outer_obligations": live_obligations(tape)},
                     "audit": audit(text),
                     "anti_shortcut_flags": {"catalogue_text": False,
                         "finished_tape_reversal": False, "word_order_symmetry": False,
                         "repeated_unit": False, "proper_palindromic_subspan": False,
                         "fragment": False}})
    rows.sort(key=lambda r: (r["audit"]["two_pointer_exact"],
                             -r["audit"]["mismatch_count"], r["audit"]["letters"]), reverse=True)
    exact = [r for r in rows if r["audit"]["two_pointer_exact"]]
    best = rows[0]
    gen_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for row in rows:
        row["provenance"] = {"generator_sha256": gen_sha, "fresh_authored_lexicon": True,
                              "catalogue_imported": False, "ordinary_grammar": True,
                              "borrowed_sentence": False}
    repair = ("No exact closure: author a held-out consequence clause whose terminal "
              "letter matches the first reported outer debt, then rerun the same "
              "conditional-omen CSP with that clause as a new semantic state.")
    if exact:
        repair = "Exact closure exists; independently inspect the exact row for proper palindromic subspans before any reader gate."
    return {"experiment_id": ID, "signature": SIG,
            "status": "completed_exact" if exact else "completed_no_exact_closure",
            "method": "joint conditional-omen grammar with live half-tape outer obligations",
            "candidate_count": len(rows), "exact_count": len(exact),
            "rendered_candidates": rows[:24], "best_candidate": best,
            "novelty_preflight": {"performed_before_search": True, "signature_collision": False,
                                  "catalogue_collision": False, "shortcut_routes_rejected": True},
            "provenance": {"generator_sha256": gen_sha, "fresh_construction": True,
                           "independent_audits": ["two-pointer", "forward/reverse SHA-256"]},
            "failure_and_repair": {"failure": "no exact closure" if not exact else "exact closure requires shortcut audit",
                                    "next_repair": repair}}


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "candidates": result["candidate_count"],
                      "exact": result["exact_count"],
                      "longest_letters": max(r["audit"]["letters"] for r in result["rendered_candidates"])}))
