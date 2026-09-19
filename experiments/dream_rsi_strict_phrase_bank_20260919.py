"""Strict Dream-RSI phrase-bank repair with a reader-facing admission ledger.

The local model proposes fresh, role-tagged phrase material; it never supplies
an accepted tape.  The exact decoder, independent audits, shared mechanical
gate, and Shakespearean repair rubric run after proposal.  This lane exists to
repair the first Dream-RSI pilot's concrete failure: it found a 66-letter exact
closure whose interior contained a hidden proper palindrome span.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import has_self_palindromic_proper_multiword_span, mechanical_admission_checks, tokenize
from server.v4 import _evaluate
from experiments import dream_rsi_model_phrase_bank_lattice_20260918 as base

EXPERIMENT_ID = "dream-rsi-strict-phrase-bank-20260919"
PROMPT = (
    "Generate 48 distinct original Shakespearean-English phrases, each 3 to 8 words, "
    "about a speaker, messenger, king, lover, book, bell, rose, river, or storm. "
    "Every phrase must be a natural clause or noun/verb phrase suitable for one coherent scene, "
    "not a list, quotation, famous palindrome, reversed-word pair, or fragment. Use ordinary "
    "lowercase ASCII words. Return JSON only as "
    '{"items":[{"text":"...","role":"subject|verb_phrase|object|adjunct|discourse"}]}.'
)


def strict_status(row: dict[str, Any]) -> dict[str, Any]:
    """Re-run the hard gate and expose model feedback as diagnostics only."""
    text = str(row["rendered"])
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=120)
    hidden = has_self_palindromic_proper_multiword_span(tokenize(text))
    audit = row["audit"]
    exact = bool(audit.get("two_pointer_exact")) and bool(audit.get("sha_equal_under_reversal"))
    evaluation = _evaluate(text)
    return {
        "exact_two_pointer_and_sha": exact,
        "hidden_proper_span": hidden,
        "mechanical_checks": checks,
        "mechanically_admitted": exact and not hidden and all(checks.values()),
        "rlaif_diagnostic": evaluation["rlaif"],
        "reader_status": "not_run; programmatic metrics never certify readability",
    }


def run(*, seeds: int = 6, min_letters: int = 39, max_letters: int = 120) -> dict[str, Any]:
    # Keep the proposal model and seed fixed for replay, but make the prompt
    # explicitly scene-oriented.  The model is not an acceptance oracle.
    base.PROMPT = PROMPT
    result = base.run(seeds=seeds, min_letters=min_letters, max_letters=max_letters)
    rows = []
    for row in result["records"]:
        row = dict(row)
        row["strict_gate"] = strict_status(row)
        rows.append(row)
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]]
    admitted = [row for row in rows if row["strict_gate"]["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "method": "Dream-RSI Shakespearean phrase proposals plus exact phrase lattice with hidden-span repair",
        "status": "completed_exact" if exact else "completed_no_exact_closure",
        "actual_candidates": rows,
        "exact_candidates": exact,
        "mechanically_admitted": admitted,
        "stats": {
            "model_phrases": result["stats"]["model_phrases"],
            "seeds": seeds,
            "exact": len(exact),
            "mechanically_admitted": len(admitted),
            "longest_exact": max((row["letters"] for row in exact), default=0),
            "longest_admitted": max((row["letters"] for row in admitted), default=0),
        },
        "failure_and_repair": {
            "prior_failure": "the 66-letter exact closure contained a hidden proper palindrome span",
            "repair": "re-run fresh scene phrase proposals and make hidden-span rejection an explicit per-row gate",
            "next": "only an admitted row enters a blinded intact-prose versus shuffled-control reader package",
        },
        "provenance": {
            "model": base.MODEL,
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "catalogue_imported": False,
            "finished_tape_reversed": False,
            "rlaif_used_for_promotion": False,
            "independent_audits": ["outside-in two-pointer", "forward/reverse SHA-256"],
        },
        "reader_gate": "closed; exactness, LM scores, and RLAIF diagnostics do not certify readability",
    }


if __name__ == "__main__":
    payload = run()
    output = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    for row in payload["exact_candidates"]:
        print(f"{row['letters']} letters | {row['rendered']} | admitted={row['strict_gate']['mechanically_admitted']}")
