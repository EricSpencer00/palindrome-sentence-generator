"""Save exact working palindromes while carrying prose debt explicitly.

This is a result-first lineage ledger, not a reader-admission gate.  The 44
letter ``now / won`` extension and the 54 letter Noel line are replayed from
independent source artifacts; the 106 letter insertion is retained as a rough
exact draft.  Each surface is re-audited here with a fresh two-pointer pass and
forward/reverse SHA-256.  Grammar and discourse debt stay visible so later
seam work can grow the longest exact line without mistaking a rough draft for
reader-worthy prose.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "working-debt-growth-20260921.json"

SEED = "An aide rips nine memos; some men inspire Diana."

CANDIDATES = [
    {
        "id": "now-won-extension",
        "text": "Now, an aide rips nine memos; some men inspire. Diana won.",
        "parent": "38-letter seed",
        "growth_letters": 6,
        "readability_track": "working_human_review_needed",
        "seam_debt": [
            "seed lineage remains visible inside the surface",
            "the discourse join before Diana is abrupt",
        ],
        "next_growth": "add a new outside-in overhang window before now and after won, keeping multiple boundary segmentations alive",
    },
    {
        "id": "noel-saga-extension",
        "text": "Was Noel an era, a gas, an item smart? Trams met in a, saga, arena, Leon saw.",
        "parent": "independent 44-letter discourse diagnostic",
        "growth_letters": 10,
        "readability_track": "rough_but_intelligible_fragments",
        "seam_debt": [
            "proper-name and question seam is syntactically uneven",
            "the middle gas/item inventory needs a coherent scene",
        ],
        "next_growth": "retain the 54-letter tape as a separate seam lineage and replace one complete semantic window jointly, never edit characters in place",
    },
    {
        "id": "involution-lexicon-draft",
        "text": "An aide rips nine memos; diaper deliver drawer stressed gateman nametag desserts reward reviled repaid some men inspire Diana.",
        "parent": "38-letter seed",
        "growth_letters": 68,
        "readability_track": "rough_exact_draft_not_reader_ready",
        "seam_debt": [
            "stacked reverse lexical units create a word-salad interior",
            "the exact tape is useful as a long debt-bearing draft only",
        ],
        "next_growth": "use the long tape only to locate repair windows; grow the 44-letter working line with ordinary clause spans instead of adding more reverse-word pairs",
    },
]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def audit(text: str) -> dict[str, object]:
    tape = normalize(text)
    mismatches = [
        (i, tape[i], tape[-1 - i])
        for i in range(len(tape) // 2)
        if tape[i] != tape[-1 - i]
    ]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "normalized": tape,
        "two_pointer_exact": bool(tape) and not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def run() -> dict[str, object]:
    seed_audit = audit(SEED)
    rows = []
    for candidate in CANDIDATES:
        row = dict(candidate)
        row["audit"] = audit(candidate["text"])
        row["provenance"] = {
            "source": "replayed authored working-result lineage",
            "finished_tape_reversal": False,
            "punctuation_changes_letters": False,
            "borrowed_catalogue_text": False,
            "reader_certified": False,
            "human_readability_evidence": "not yet run; preserve for blinded intact-vs-shuffled study only after seam repair",
        }
        row["exact_closure"] = bool(
            row["audit"]["two_pointer_exact"] and row["audit"]["sha_equal"]
        )
        rows.append(row)
    return {
        "experiment_id": "working-debt-growth-20260921",
        "method": "exact working-result ledger with explicit character/discourse debt and seam ownership",
        "seed_regression": {"text": SEED, "audit": seed_audit},
        "stats": {
            "working_candidates": len(rows),
            "exact_working_candidates": sum(row["exact_closure"] for row in rows),
            "longest_exact_working_letters": max(row["audit"]["letters"] for row in rows),
            "longest_intelligible_track_letters": max(
                row["audit"]["letters"]
                for row in rows
                if row["readability_track"] != "rough_exact_draft_not_reader_ready"
            ),
            "growth_over_seed": max(row["audit"]["letters"] for row in rows) - seed_audit["letters"],
        },
        "working_candidates": rows,
        "best_intelligible_track": rows[1],
        "longest_rough_track": rows[2],
        "reader_gate": "closed; these are working drafts, not reader-validated outputs",
        "next_growth": "Run the proven two-sided overhang engine from the 44-letter now/won line with several boundary segmentations; retain exact closures and carry discourse debt instead of scoring only a frozen prose bank.",
        "independent_audits": ["fresh two-pointer mismatch scan", "forward/reverse SHA-256"],
    }


if __name__ == "__main__":
    payload = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    for row in payload["working_candidates"]:
        print(f"{row['audit']['letters']} letters | {row['text']} | exact={row['exact_closure']}")
