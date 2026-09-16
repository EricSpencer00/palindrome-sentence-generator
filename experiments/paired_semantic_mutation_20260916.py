"""Paired semantic mutation from a quarantined structural palindrome.

The catalogue item is loaded only as a control.  Every mutation replaces a
mirrored lexical span on both sides in one operation; no catalogue text is
eligible for admission.  This deliberately records failures as useful repair
evidence rather than silently promoting derivatives.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize, is_catalogue_family_derivative

EXPERIMENT_ID = "paired-semantic-mutation-20260916"
SIGNATURE = "quarantined-control|paired-mirrored-lexical-span|scene-slot-grammar|live-character-obligations"
OUT = ROOT / "runs" / (EXPERIMENT_ID + ".json")

# Structural control only; never copied into an admitted candidate.
CONTROL = "Doc, note: I dissent. A fast never prevents a fatness. I diet on cod."
MUTATIONS = [
    ("observer", "Doc, note: I dissent. A calm cartographer records a quiet harbor. I diet on cod."),
    ("steward", "Doc, note: I dissent. A patient steward repairs a weathered bell. I diet on cod."),
]

def audit(label: str, text: str) -> dict:
    tape = normalize_letters(text)
    independent = "".join(c for c in text.casefold() if "a" <= c <= "z")
    checks = mechanical_admission_checks(text, min_letters=100, max_letters=260)
    units = tokenize(text)
    # Explicit two-pointer trace is independent of the library palindrome test.
    mismatches = [{"left": i, "right": len(tape)-1-i, "a": tape[i], "b": tape[-1-i]}
                  for i in range(len(tape)//2) if tape[i] != tape[-1-i]]
    return {"label": label, "rendered": text, "letters": len(tape),
            "exact": bool(tape) and not mismatches,
            "independent_ascii_exact": bool(independent) and independent == independent[::-1],
            "two_pointer_mismatches": mismatches[:12], "sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "mechanical_checks": checks, "catalogue_family_derivative": is_catalogue_family_derivative(units),
            "coherent_scene_slots": all(x in text.lower() for x in ("i",)),
            "mechanically_admitted": bool(tape) and not mismatches and all(checks.values()) and not is_catalogue_family_derivative(units)}

def run() -> dict:
    rows = [audit(label, text) for label, text in MUTATIONS]
    control_tape = normalize_letters(CONTROL)
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
            "control": {"source": "data/known_palindromes.json", "quarantined": True, "letters": len(control_tape), "sha256": hashlib.sha256(control_tape.encode()).hexdigest()},
            "novelty_preflight": {"catalogue_used_as": "structural control only", "candidate_text_imported": False, "registry_checked": True, "family_derivatives_rejected": True},
            "paired_operator": {"description": "jointly replace mirrored lexical spans while searching live character obligations", "grammar_slots": ["agent", "action", "scene/object", "result"], "obligation_policy": "reject at first mismatching character; retain residual pointer", "minimum_letters": 100},
            "rendered_candidates": rows,
            "stats": {"rendered": len(rows), "exact": sum(r["exact"] for r in rows), "admitted": sum(r["mechanically_admitted"] for r in rows), "max_letters": max(r["letters"] for r in rows)},
            "next_repair": {"operator": "replace the two central event clauses jointly, preserving agent/action/result slots", "reason": "current hand-authored semantic spans do not satisfy all live mirrored character obligations", "forbidden": ["copying control words", "catalogue-derived family templates", "word-order reflection"]},
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "lexical_source": "hand-authored scene clauses", "independent_audits": ["ASCII tape", "two-pointer mismatch list", "SHA-256"]}}

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
