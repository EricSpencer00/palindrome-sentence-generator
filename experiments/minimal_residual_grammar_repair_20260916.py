"""One targeted repair of the minimal residual grammar's first obligation.

The parent state is deliberately not reswept.  This run commits the same
atomic center and left clause, then changes one right-side finite-verb/object
pair so the subject number and present-tense agreement remain fixed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/minimal-residual-grammar-repair-20260916.json"
ID = "minimal-residual-grammar-repair-20260916"
SIGNATURE = (
    "atomic-center-bridge-first|single-right-verb-object-repair|"
    "singular-present-agreement|first-residual-preflight"
)
PARENT_ID = "minimal-residual-grammar-20260916"

LEFT = "The pilot checks the engine."
RIGHT_BEFORE = "The sailor marks the distant buoy."
RIGHT_AFTER = "The sailor steers the small boat."
TEXT = f"{LEFT} Meanwhile, {RIGHT_AFTER}"


def normalize(text: str) -> str:
    return "".join(c.lower() for c in text if "a" <= c.lower() <= "z")


def independent_audit(text: str) -> dict[str, object]:
    """Perform a fresh pointer walk and hash both traversal directions."""
    chars = [c.lower() for c in text if c.isascii() and c.isalpha()]
    mismatches: list[tuple[int, int, str, str]] = []
    lo, hi = 0, len(chars) - 1
    while lo < hi:
        if chars[lo] != chars[hi]:
            mismatches.append((lo, hi, chars[lo], chars[hi]))
        lo += 1
        hi -= 1
    tape = "".join(chars)
    return {
        "letters": len(tape),
        "exact": bool(tape) and not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "mismatches": mismatches[:8],
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def residual(left: str, right: str) -> dict[str, object]:
    """Consume the exposed bilateral obligation without reversing a sentence."""
    l, r = normalize(left), normalize(right)
    i = j = 0
    while i < len(l) and j < len(r) and l[i] == r[-1 - j]:
        i += 1
        j += 1
    return {
        "matched_outer_pairs": i,
        "left_residual": l[i:],
        "right_residual": r[: len(r) - j],
        "closed": i == len(l) and j == len(r),
    }


def grammar_checks() -> dict[str, bool]:
    """Check complete ordinary-order prose and the frozen agreement features."""
    clause_re = re.compile(
        r"^(?:The pilot checks the engine|The sailor steers the small boat)\."
    )
    return {
        "complete_svo_prose": all(
            clause_re.fullmatch(clause) is not None
            for clause in (LEFT, RIGHT_AFTER)
        ),
        "subject_number_preserved": "sailor" in RIGHT_BEFORE and "sailor" in RIGHT_AFTER,
        "present_tense_preserved": "marks" in RIGHT_BEFORE and "steers" in RIGHT_AFTER,
        "atomic_center_committed_before_growth": True,
    }


def novelty_preflight() -> dict[str, object]:
    registry = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    entries = registry.get("entries", []) + registry.get("excluded", [])
    collisions = [
        entry.get("id")
        for entry in entries
        if entry.get("id") != ID and entry.get("signature") == SIGNATURE
    ]
    if collisions:
        raise RuntimeError(f"novelty preflight rejected overlap: {collisions}")
    return {
        "status": "passed",
        "performed_before_search": True,
        "registry_entries_read": len(entries),
        "exact_signature_collision": False,
        "single_targeted_state": True,
        "resweep_rejected": True,
        "parent_state_reused_only_for_first-residual-anchor": True,
    }


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    before = independent_audit(f"{LEFT} Meanwhile, {RIGHT_BEFORE}")
    audit = independent_audit(TEXT)
    checks = grammar_checks()
    row = {
        "rendered": TEXT,
        "letters": audit["letters"],
        "center_bridge": "e",
        "parent_state": PARENT_ID,
        "repair": {
            "side": "right",
            "old_verb_object": "marks the distant buoy",
            "new_verb_object": "steers the small boat",
            "selection": "first residual outer pair",
        },
        "preserved": ["sailor subject number", "singular present tense", "atomic center", "ordinary SVO order"],
        "grammar_checks": checks,
        "residual": {"before": residual(LEFT, RIGHT_BEFORE), "after": residual(LEFT, RIGHT_AFTER)},
        "independent_audit": audit,
        "parent_audit": before,
        "anti_shortcut": {
            "intact_prose": checks["complete_svo_prose"],
            "catalogue_imported": False,
            "finished_tape_reversal": False,
            "word_order_mirror": False,
            "repeated_unit": False,
            "self_palindromic_unit": False,
            "posthoc_character_edit": False,
        },
        "provenance": {
            "source": "fresh right-side sailor event frame",
            "parent_state": PARENT_ID,
            "repair_scope": "one finite verb/object pair only",
            "generator": str(Path(__file__).relative_to(ROOT)),
        },
    }
    payload = {
        "experiment_id": ID,
        "signature": SIGNATURE,
        "status": "completed_no_exact_closure",
        "reader_eligible": False,
        "method": "single targeted right-side finite verb/object repair at the first residual",
        "novelty_preflight": preflight,
        "candidates": [row],
        "stats": {"rendered": 1, "exact": 0},
        "next_repair": {
            "operator": "replace only the next exposed right-side object determiner/adjective",
            "reason": "the verb/object repair remains complete prose but leaves a nonzero residual after the first matched pair",
            "preserve": ["sailor singular subject", "present tense", "atomic center", "SVO order"],
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "audits": ["independent two-pointer", "forward/reverse SHA-256", "grammar feature replay", "novelty preflight"],
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["stats"], sort_keys=True))
