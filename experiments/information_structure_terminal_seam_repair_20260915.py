"""Repair the information-structure route by mutating result-clause endings.

The prior route kept complete negative cause/result plans but did not expose a
useful character seam.  This repair keeps the focus/presupposition/polarity
state fixed and searches authored, semantically equivalent result clauses with
different terminal lexicalizations.  It is deliberately a repair of that
family, not a new cross-product claim.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ROOT = Path(__file__).resolve().parents[1]
FAMILY = "information-structure-focus-scope"
SIGNATURE = (
    "information-structure-focus|presupposition-scope|"
    "polarity-preserving-terminal-seam-repair|causal-result-lexicalization|"
    "independent-two-pointer-audit"
)

PLANS = [
    {
        "left": "The careful editor did not revise the draft because the facts changed.",
        "rights": [
            "The facts changed, so the careful editor left the draft alone.",
            "The facts changed, so the careful editor set the draft aside.",
            "The facts changed, so the careful editor filed the draft away.",
            "Because the facts changed, the careful editor shelved the draft.",
        ],
    },
    {
        "left": "The patient teacher did not grade the essay because the class waited.",
        "rights": [
            "The class waited, so the patient teacher left the essay ungraded.",
            "The class waited, so the patient teacher set the essay aside.",
            "The class waited, so the patient teacher held the essay back.",
            "Because the class waited, the patient teacher shelved the essay.",
        ],
    },
    {
        "left": "The quiet farmer did not harvest the grain because the clouds gathered.",
        "rights": [
            "The clouds gathered, so the quiet farmer left the grain unharvested.",
            "The clouds gathered, so the quiet farmer set the grain aside.",
            "The clouds gathered, so the quiet farmer delayed the harvest.",
            "Because the clouds gathered, the quiet farmer spared the grain.",
        ],
    },
    {
        "left": "The skilled baker did not sell the bread because the market closed.",
        "rights": [
            "The market closed, so the skilled baker left the bread unsold.",
            "The market closed, so the skilled baker set the bread aside.",
            "The market closed, so the skilled baker held the bread back.",
            "Because the market closed, the skilled baker kept the bread.",
        ],
    },
]


def two_pointer(text: str) -> bool:
    tape = normalize_letters(text)
    if not tape:
        return False
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            return False
        i += 1
        j -= 1
    return True


def row(left: str, right: str, plan_index: int, variant_index: int) -> dict:
    text = f"{left} {right}"
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=260)
    outer = 0
    for a, b in zip(tape, reversed(tape)):
        if a != b:
            break
        outer += 1
    return {
        "plan_index": plan_index,
        "variant_index": variant_index,
        "rendered": text,
        "letters": len(tape),
        "outer_matching_pairs": outer,
        "exact": bool(tape) and tape == tape[::-1],
        "independent_two_pointer": two_pointer(text),
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "tokens": list(tokenize(text)),
        "admitted": all(checks.values()),
        "failed_checks": [key for key, value in checks.items() if not value],
        "readable_status": "diagnostic_only",
        "semantic_state": {
            "focus": "agent action",
            "presupposition": "cause event",
            "polarity": "negative-to-result",
            "repair": "terminal result lexicalization",
        },
    }


def main() -> None:
    rows = [
        row(plan["left"], right, pi, ri)
        for pi, plan in enumerate(PLANS)
        for ri, right in enumerate(plan["rights"])
    ]
    output = {
        "status": "repair_of_registered_family",
        "family": FAMILY,
        "signature": SIGNATURE,
        "preflight": {
            "registry_entries": 64,
            "excluded_families": 5,
            "manual_review_required": True,
            "overlap": FAMILY,
            "disposition": "repair, not a retained family",
        },
        "operator": "fixed information structure with authored terminal result lexicalizations",
        "rows": rows,
        "independent_audit": {
            "method": "explicit opposing-index scan over every rendered probe",
            "probes_checked": len(rows),
            "primary_exact": sum(item["exact"] for item in rows),
            "independent_exact": sum(item["independent_two_pointer"] for item in rows),
            "disagreements": [item["rendered"] for item in rows if item["exact"] != item["independent_two_pointer"]],
        },
        "exact_survivors": [item for item in rows if item["exact"]],
        "admitted_survivors": [item for item in rows if item["exact"] and item["admitted"]],
        "reader_status": "not run; no admitted survivor",
        "next_repair": "hold semantic roles fixed and introduce a controlled connective/tense alternation at the causal seam; no bank-only replay",
        "provenance": {
            "source": "four authored negative cause/result plans and four authored result lexicalizations per plan",
            "generator": str(Path(__file__).relative_to(ROOT)),
        },
    }
    out = ROOT / "runs/information-structure-terminal-seam-repair-2026-09-15.json"
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "exact": len(output["exact_survivors"]), "admitted": len(output["admitted_survivors"]), "best_outer_pairs": max(item["outer_matching_pairs"] for item in rows)}))


if __name__ == "__main__":
    main()
