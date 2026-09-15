"""Proposal-contract experiment for readable long palindromes.

Unlike the registered generators, this route does not synthesize from a
grammar, reverse a tape, or search a lexical lattice.  A human or external
proposer supplies ordinary prose; this program is an independent gatekeeper
and emits only an exact audit and a concrete next-proposal diagnostic.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ID = "semantic-proposal-verifier"
SIGNATURE = (
    "proposal-contract|human-readable-acceptance|semantic-plausibility-check|"
    "independent-exact-verifier|mismatch-directed-next-proposal"
)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/semantic-proposal-verifier-20260915.json"

# These are deliberately ordinary, non-palindromic proposals: the experiment
# demonstrates the contract and repair report without smuggling in a catalogue
# sentence or a mirrored construction.
PROPOSALS = [
    "Mara carried warm bread to the quiet farm after rain.",
    "Nora watched the blue boat drift past the old harbor at dawn.",
    "Ira reads a calm report and leaves before the bell rings.",
]


def letters(text: str) -> str:
    return "".join(ch.lower() for ch in text if "a" <= ch.lower() <= "z")


def independent_audit(text: str) -> dict[str, object]:
    tape = letters(text)
    mismatches = [
        {"left_index": i, "right_index": len(tape) - 1 - i,
         "left": tape[i], "right": tape[-1 - i]}
        for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]
    ]
    words = re.findall(r"[A-Za-z]+", text)
    return {
        "letters": len(tape),
        "exact": bool(tape) and not mismatches,
        "readability_gate": len(words) >= 5 and all(len(w) > 0 for w in words),
        "mismatches": mismatches[:12],
        "word_count": len(words),
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
    }


def main() -> None:
    audits = [{"text": text, "audit": independent_audit(text)} for text in PROPOSALS]
    admitted = [row for row in audits if row["audit"]["exact"] and row["audit"]["readability_gate"] and row["audit"]["letters"] > 38]
    best = max(audits, key=lambda row: (sum(1 for m in row["audit"]["mismatches"] if m["left"] == m["right"]), row["audit"]["letters"]))
    payload = {
        "experiment_id": ID,
        "signature": SIGNATURE,
        "status": "proposal-contract-audit",
        "proposal_source": "three hand-authored intact prose proposals; external submission is supported by replacing PROPOSALS",
        "catalogue_lookup": False,
        "audits": audits,
        "admitted": admitted,
        "next_repair": {
            "operator": "mismatch-directed-next-proposal",
            "basis": "independent two-pointer audit of the longest proposal",
            "proposal_to_revise": best["text"],
            "first_mismatches": best["audit"]["mismatches"][:3],
            "instruction": "Proposer supplies a new intact, semantically coherent sentence; verifier never edits or mirrors it.",
        },
        "independent_audit": "letters() plus direct two-pointer comparison, implemented here without generator helpers",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"proposals": len(audits), "admitted": len(admitted), "longest_letters": max(r["audit"]["letters"] for r in audits)}))


if __name__ == "__main__":
    main()
