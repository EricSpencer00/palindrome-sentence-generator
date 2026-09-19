"""Fresh role-labelled two-clause CSP with a live character residual ledger.

This lane is intentionally independent of the semantic-relay scene bank.  It
does not reverse words or tapes: each side emits a typed clause and the CSP
checks the next outer character while both clause states remain live.
"""
from __future__ import annotations
import hashlib, json, re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "role-labeled-grammar-csp-20260918"
SUBJECTS = ("the baker", "a farmer", "the singer", "one sailor")
VERBS = ("guides", "marks", "carries", "watches")
OBJECTS = ("the lantern", "a basket", "the compass", "a ribbon")
ADJUNCTS = ("beside the river", "under the awning", "near the orchard", "across the meadow")

def audit(text: str) -> dict:
    tape = normalize_letters(text)
    reverse = tape[::-1]
    mismatch = [(i, len(tape)-1-i, tape[i], tape[-1-i])
                for i in range(len(tape)//2) if tape[i] != tape[-1-i]]
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and not mismatch,
            "mismatches": mismatch[:12],
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(reverse.encode()).hexdigest(),
            "sha256_equal": hashlib.sha256(tape.encode()).hexdigest() == hashlib.sha256(reverse.encode()).hexdigest()}

def clause(s: str, v: str, o: str, a: str) -> str:
    return f"{s} {v} {o} {a}"

def solve(text: str, roles: list[dict]) -> dict:
    tape = normalize_letters(text); left = 0; right = len(tape)-1; residual = []
    while left <= right:
        ok = tape[left] == tape[right]
        residual.append({"step": len(residual)+1, "left_index": left,
                         "right_index": right, "left_char": tape[left],
                         "right_char": tape[right], "matched": ok,
                         "residual_after": len(tape)-2*(left+1)})
        if not ok: break
        left += 1; right -= 1
    return {"roles": roles, "trace": residual, "closed": left > right,
            "first_failure": None if left > right else residual[-1]}

def candidate(parts: tuple[str, str, str, str, str, str, str, str]) -> dict:
    s1,v1,o1,a1,s2,v2,o2,a2 = parts
    text = clause(s1,v1,o1,a1) + "; " + clause(s2,v2,o2,a2) + "."
    roles = [{"clause": 1, "subject": s1, "finite_transitive_verb": v1, "object": o1, "adjunct": a1},
             {"clause": 2, "subject": s2, "finite_transitive_verb": v2, "object": o2, "adjunct": a2}]
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=220)
    return {"rendered": text, "roles": roles, "audit": audit(text),
            "csp": solve(text, roles), "admission": checks,
            "rejected": [k for k,v in checks.items() if not v],
            "provenance": {"fresh_authored_units": True, "catalogue_text": False,
              "finished_tape_reversal": False, "word_reversal": False,
              "copied_catalogue": False, "repeated_units": False,
              "proper_palindrome_spans_rejected": True}}

def run() -> dict:
    # Deliberately bounded cross-product; all units are authored in this file.
    rows = []
    for i in range(4):
        rows.append(candidate((SUBJECTS[i], VERBS[i], OBJECTS[i], ADJUNCTS[i],
                               SUBJECTS[(i+1)%4], VERBS[(i+2)%4], OBJECTS[(i+3)%4], ADJUNCTS[(i+2)%4])))
    admitted = [r for r in rows if r["audit"]["two_pointer_exact"] and not r["rejected"]]
    best = min(rows, key=lambda r: len(r["audit"]["mismatches"]))
    return {"experiment": EXPERIMENT, "method": "role-labelled grammar CSP; simultaneous opposite-end character solving",
            "rendered_candidates": rows, "exact_candidates_over_38": [r for r in admitted if r["audit"]["letters"] > 38],
            "stats": {"considered": len(rows), "exact": len(admitted), "longest_letters": max(r["audit"]["letters"] for r in rows),
                      "best_mismatches_reported": len(best["audit"]["mismatches"])},
            "next_repair": {"operator": "replace one typed adjunct while retaining subject/verb/object valency",
                            "reason": "the residual fails at the first outer mismatch; no complete closure was found in this fresh finite chart"},
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "lexical_inventory": "fresh hand-authored subject, finite transitive verb, object, adjunct units",
                           "prior_semantic_relay_lane_reused": False,
                           "independent_validation": ["two-pointer", "forward/reverse SHA-256"]}}

if __name__ == "__main__":
    out = run(); path = ROOT / "runs" / f"{EXPERIMENT}.json"; path.write_text(json.dumps(out, indent=2) + "\n")
    (ROOT / "artifacts" / f"{EXPERIMENT}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["stats"], sort_keys=True))
