"""Word-internal seam equations coupled to independently authored prose.

The search chooses one internal boundary in each selected word (for example,
``re|corded``), then constrains the boundary-adjacent characters across two
complete ordinary clauses.  It never builds a tape first or reverses words.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / "word-internal-seam-equation-20260916.json"
EXPERIMENT_ID = "word-internal-seam-equation-20260916"
SIGNATURE = (
    "word-internal-seam-equation|morpheme-boundary-character-pairing|"
    "independent-complete-clauses|ordinary-word-order|live-seam-obligation|"
    "independent-pointer-sha-audit"
)
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


# The words are selected as semantic roles, not as reversible word pairs.
LEFT = (
    ("the", "curator", "curate", "the curator recorded a repair ledger beside the quiet archive"),
    ("a", "teacher", "teach", "a teacher guided the patient apprentice through the winter archive"),
    ("the", "reworker", "rework", "the reworker marked a careful revision under the copper lamp"),
)
RIGHT = (
    ("the", "observer", "observe", "the observer reported a measured signal near the eastern station"),
    ("a", "rewriter", "rewrite", "a rewriter carried the revised message toward the harbor office"),
    ("the", "recorder", "record", "the recorder preserved a precise account beside the weathered map"),
)


def novelty_preflight() -> dict[str, object]:
    registry = json.loads(REGISTRY.read_text())
    rows = [row for row in registry.get("entries", []) if row.get("id") != EXPERIMENT_ID]
    overlaps = sorted({row.get("signature") for row in rows if row.get("signature") == SIGNATURE})
    artifact = str(Path(__file__).relative_to(ROOT))
    collisions = [row.get("artifact") for row in rows if row.get("artifact") == artifact]
    forbidden = ("fixed-tape", "reverse-segmentation", "word-order-mirror", "broad morphology")
    route = "word-internal seam equation; one boundary per independently authored clause"
    result = {
        "status": "passed" if not overlaps and not collisions else "blocked",
        "registry_entries_before_run": len(registry.get("entries", [])),
        "signature_overlaps": overlaps,
        "artifact_collisions": collisions,
        "route": route,
        "rejected_routes": list(forbidden),
    }
    if result["status"] != "passed":
        raise RuntimeError(f"novelty preflight blocked: {result}")
    return result


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    independent = "".join(ch.lower() for ch in text if "a" <= ch.lower() <= "z")
    i, j = 0, len(independent) - 1
    while i < j and independent[i] == independent[j]:
        i += 1
        j -= 1
    checks = mechanical_admission_checks(text, min_letters=100, max_letters=240)
    return {
        "letters": len(tape),
        "normalized_tape": tape,
        "independent_ascii_tape": independent,
        "exact": bool(independent) and independent == independent[::-1],
        "independent_two_pointer_exact": bool(independent) and i >= j,
        "first_mismatch": None if i >= j else {"left_index": i, "right_index": j, "left": independent[i], "right": independent[j]},
        "sha256_forward": hashlib.sha256(independent.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(independent[::-1].encode()).hexdigest(),
        "mechanical_checks": checks,
    }


def seam(word: str, split: int) -> dict[str, object]:
    return {"word": word, "split": split, "left_morpheme": word[:split], "right_morpheme": word[split:],
            "left_terminal": word[split - 1], "right_initial": word[split]}


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    probes = []
    # Rendering begins only after preflight.  The split itself is the live state.
    for left, right in itertools.product(LEFT, RIGHT):
        left_word, right_word = left[1], right[1]
        # Keep the boundary internal even when ``-er`` is word-final.
        left_split = left_word.index("er") + 1 if "er" in left_word else max(1, len(left_word) // 2)
        right_split = right_word.index("er") + 1 if "er" in right_word else max(1, len(right_word) // 2)
        text = f"{left[3]}; {right[3]}."
        lseam, rseam = seam(left_word, left_split), seam(right_word, right_split)
        seam_match = lseam["left_terminal"] == rseam["left_terminal"]
        probes.append({
            "text": text,
            "semantic_roles": ["agent", "event", "patient", "setting"],
            "seams": {"left": lseam, "right": rseam},
            "seam_equation": f"{lseam['left_terminal']} == {rseam['left_terminal']}",
            "seam_match": seam_match,
            "audit": audit(text),
            "anti_shortcut_flags": {"fixed_tape": False, "reverse_segmentation": False, "word_order_mirror": False,
                                     "repeated_unit": False, "catalogue_lookup": False, "posthoc_reversal": False},
        })
    probes.sort(key=lambda row: (row["audit"]["exact"], row["seam_match"], row["audit"]["letters"]), reverse=True)
    best = probes[0]
    return {
        "family": "word-internal-seam-equation",
        "target_letters": 100,
        "novelty_preflight": preflight,
        "candidate_count": len(probes),
        "best": best,
        "rendered_intact_scene": best["text"],
        "provenance": {"authored_inventory": True, "catalogue_lookup": False, "known_seeds": False,
                       "independent_clause_authoring": True, "word_order_mirror": False},
        "next_repair": "replace only the two seam-adjacent morphemes at the first mismatch while preserving each clause's agent, event, patient, and setting roles",
    }


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
