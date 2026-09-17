"""Targeted repair for the first residual of the word-internal seam lane.

Only the two morphemes adjacent to the first global mismatch are replaced:
the left determiner and the right clause's patient noun.  Clause roles and
ordinary word order remain fixed; the already-satisfied internal ``e == e``
seam is retained as a diagnostic rather than used as a tape.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / "word-internal-seam-first-mismatch-repair-20260916.json"
EXPERIMENT_ID = "word-internal-seam-first-mismatch-repair-20260916"
SIGNATURE = (
    "word-internal-seam-first-mismatch-repair|two-adjacent-morpheme-edit|"
    "role-preserving-complete-prose|independent-pointer-sha"
)

sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


SOURCE = "a teacher guided the patient apprentice through the winter archive; the recorder preserved a precise account beside the weathered map."
REPAIRED = "the teacher guided the patient apprentice through the winter archive; the recorder preserved a precise account beside the weathered chart."


def novelty_preflight() -> dict[str, object]:
    registry = json.loads(REGISTRY.read_text())
    entries = [row for row in registry.get("entries", []) if row.get("id") != EXPERIMENT_ID]
    overlaps = sorted(row.get("signature") for row in entries if row.get("signature") == SIGNATURE)
    artifact = str(Path(__file__).relative_to(ROOT))
    collisions = [row.get("artifact") for row in entries if row.get("artifact") == artifact]
    duplicate_sweep = any("word-internal-seam-first-mismatch-repair" in str(row.get("id")) for row in entries)
    result = {
        "status": "passed" if not overlaps and not collisions and not duplicate_sweep else "blocked",
        "registry_entries_before_run": len(registry.get("entries", [])),
        "signature_overlaps": overlaps,
        "artifact_collisions": collisions,
        "duplicate_sweep": duplicate_sweep,
        "route": "first-global-mismatch repair; two seam-adjacent morphemes only",
        "rejected_routes": ["duplicate sweep", "fixed tape", "posthoc reversal", "broad morphology"],
    }
    if result["status"] != "passed":
        raise RuntimeError(f"novelty preflight blocked: {result}")
    return result


def audit(text: str) -> dict[str, object]:
    normalized = normalize_letters(text)
    independent = "".join(ch.lower() for ch in text if "a" <= ch.lower() <= "z")
    i, j = 0, len(independent) - 1
    while i < j and independent[i] == independent[j]:
        i += 1
        j -= 1
    return {
        "letters": len(normalized),
        "normalized_tape": normalized,
        "independent_ascii_tape": independent,
        "exact": bool(independent) and independent == independent[::-1],
        "independent_two_pointer_exact": bool(independent) and i >= j,
        "first_mismatch": None if i >= j else {"left_index": i, "right_index": j, "left": independent[i], "right": independent[j]},
        "sha256_forward": hashlib.sha256(independent.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(independent[::-1].encode()).hexdigest(),
        "mechanical_checks": mechanical_admission_checks(text, min_letters=100, max_letters=240),
    }


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    source_audit = audit(SOURCE)
    repaired_audit = audit(REPAIRED)
    return {
        "family": "word-internal-seam-first-mismatch-repair",
        "novelty_preflight": preflight,
        "source": {"text": SOURCE, "audit": source_audit},
        "rendered_intact_scene": REPAIRED,
        "repair": {
            "operator": "replace only two seam-adjacent morphemes at first global mismatch",
            "changed_morphemes": [
                {"side": "left", "role": "determiner", "before": "a", "after": "the"},
                {"side": "right", "role": "patient", "before": "map", "after": "chart"},
            ],
            "preserved_roles": ["agent", "event", "patient", "setting"],
            "preserved_internal_seam": "teacher/recorder: e == e",
            "changed_word_count": 2,
        },
        "audit": repaired_audit,
        "anti_shortcut_flags": {"fixed_tape": False, "reverse_segmentation": False, "word_order_mirror": False,
                                 "repeated_unit": False, "catalogue_lookup": False, "posthoc_reversal": False,
                                 "duplicate_sweep": False},
        "provenance": {"authored_repair": True, "catalogue_lookup": False, "known_seeds": False,
                       "independent_clause_authoring": True, "source_run": "runs/word-internal-seam-equation-20260916.json"},
        "next_repair": "replace only the next first-residual character-bearing morphemes while preserving both clauses' four semantic roles",
    }


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    print(OUT)
