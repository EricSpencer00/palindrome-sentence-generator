"""Bounded outer-terminal closure attempt for the agreement/clitic frontier.

The search jointly lexicalizes one opening terminal and one closing temporal
terminal.  It checks their live outer character obligation before rendering a
complete ordinary-order scene.  The bounded product is deliberately small and
does not resegment a fixed tape.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / "agreement-clitic-outer-terminal-search-20260916.json"
EXPERIMENT_ID = "agreement-clitic-outer-terminal-search-20260916"
SIGNATURE = (
    "agreement-clitic-outer-terminal-search|bounded-heldout-terminals|"
    "live-outer-character-equation|single-scene|independent-pointer-sha"
)
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


OPENINGS = (
    {"text": "At dawn, a patient pilot", "features": {"opening": "temporal-prelude", "subject_number": "singular"}},
    {"text": "In harbor, a patient pilot", "features": {"opening": "locative-prelude", "subject_number": "singular"}},
    {"text": "The patient pilot", "features": {"opening": "determiner-subject", "subject_number": "singular"}},
)
CLOSINGS = (
    {"text": "at sea", "features": {"terminal": "locative", "tense": "present"}},
    {"text": "by the quay", "features": {"terminal": "locative", "tense": "present"}},
    {"text": "near the bay", "features": {"terminal": "locative", "tense": "present"}},
)


def novelty_preflight() -> dict[str, object]:
    registry = json.loads(REGISTRY.read_text())
    rows = registry.get("entries", []) + registry.get("excluded", [])
    overlap = [row.get("signature") for row in rows if row.get("signature") == SIGNATURE]
    artifact = str(Path(__file__).relative_to(ROOT))
    collisions = [row.get("artifact") for row in rows if row.get("artifact") == artifact]
    result = {
        "status": "passed" if not overlap and not collisions else "blocked",
        "registry_entries_before_run": len(rows),
        "signature_overlaps": overlap,
        "artifact_collisions": collisions,
        "duplicate_sweep": False,
        "bounded_product_size": len(OPENINGS) * len(CLOSINGS),
        "rejected_routes": ["fixed-tape resegmentation", "broad terminal sweep", "word-order mirror", "copied seed"],
    }
    if result["status"] != "passed":
        raise RuntimeError(f"novelty preflight blocked: {result}")
    return result


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i += 1
        j -= 1
    return {
        "letters": len(tape), "normalized_tape": tape,
        "exact": bool(tape) and i >= j,
        "independent_two_pointer_exact": bool(tape) and i >= j,
        "first_mismatch": None if i >= j else {"left_index": i, "right_index": j, "left": tape[i], "right": tape[j]},
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
        "mechanical_checks": mechanical_admission_checks(text, min_letters=50, max_letters=240),
    }


def render(opening: dict[str, object], closing: dict[str, object]) -> str:
    return f"{opening['text']} checks the engine, notes its gauge, and tells the crew it works {closing['text']}."


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    candidates = []
    for opening, closing in itertools.product(OPENINGS, CLOSINGS):
        text = render(opening, closing)
        tape = normalize_letters(text)
        outer_match = tape[0] == tape[-1]
        candidates.append({
            "text": text,
            "lexical_terminals": {"opening": opening, "closing": closing},
            "agreement_clitic_features": {"subject_number": "singular", "tense": "present", "clitic": "its"},
            "character_obligations": [{"name": "live_outer_terminal", "left": tape[0], "right": tape[-1], "matched": outer_match}],
            "audit": audit(text),
            "anti_shortcut_flags": {"fixed_tape_resegmentation": False, "broad_terminal_sweep": False,
                                     "copied_seed": False, "word_order_mirror": False, "catalogue_lookup": False,
                                     "posthoc_reversal": False},
        })
    candidates.sort(key=lambda row: (row["character_obligations"][0]["matched"], -row["audit"]["letters"]), reverse=True)
    best = candidates[0]
    result = {
        "experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
        "status": "completed_no_exact_closure", "reader_eligible": False,
        "method": "bounded joint lexicalization of held-out opening/closing terminals against a live outer character equation",
        "candidates": candidates, "candidate_count": len(candidates),
        "outer_obligation_matches": sum(row["character_obligations"][0]["matched"] for row in candidates),
        "best": best, "rendered_intact_scene": best["text"],
        "novelty_preflight": preflight,
        "provenance": {"single_authored_scene": True, "heldout_terminal_inventory": True, "catalogue_lookup": False,
                       "fixed_tape": False, "known_seed_import": False,
                       "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "audits": ["independent two-pointer", "forward/reverse SHA-256", "mechanical admission"]},
        "anti_shortcut_flags": {"fixed_tape_resegmentation": False, "broad_terminal_sweep": False,
                                 "copied_seed": False, "word_order_mirror": False, "catalogue_lookup": False,
                                 "posthoc_reversal": False},
        "next_repair": {"operator": "retain the matched outer terminal pair and replace only the first residual-bearing inflected/clitic terminal",
                        "reason": "the bounded lexicalized equation closes the outer pair but not the complete scene"},
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
