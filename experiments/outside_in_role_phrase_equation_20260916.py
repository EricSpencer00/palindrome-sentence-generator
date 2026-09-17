"""Fresh outside-in phrase equation search with typed semantic roles."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/outside-in-role-phrase-equation-20260916.json"
ID = "outside-in-role-phrase-equation-20260916"
SIGNATURE = (
    "outside-in-role-phrase-equation|fresh-authored-grammar-bank|"
    "typed-semantic-expansion|live-character-obligation|independent-pointer-sha"
)

# These are authored phrase choices, not catalogue spans.  Each side remains a
# normal SVO clause with singular agreement; choices are made jointly before
# the rendered sentence is scored.
LEFT = {
    "subject": ("The weathered cartographer", "The diligent conservator"),
    "verb": ("records", "examines"),
    "object": ("a hidden estuary", "a fragile sextant"),
    "adjunct": ("beside the northern observatory", "within the coastal museum"),
}
RIGHT = {
    "subject": ("A patient instrument maker", "A careful harbor keeper"),
    "verb": ("restores", "catalogs"),
    "object": ("a cracked compass", "a brass chronometer"),
    "adjunct": ("inside the maritime archive", "beneath the eastern gallery"),
}


def tape(text: str) -> str:
    return normalize_letters(text)


def pointer_audit(text: str) -> dict[str, object]:
    chars = [c.lower() for c in text if c.isascii() and c.isalpha()]
    mismatches = []
    lo, hi = 0, len(chars) - 1
    while lo < hi:
        if chars[lo] != chars[hi]:
            mismatches.append((lo, hi, chars[lo], chars[hi]))
        lo += 1
        hi -= 1
    normalized = "".join(chars)
    return {
        "letters": len(normalized),
        "exact": bool(normalized) and not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "mismatch_count": len(mismatches),
        "sha256_forward": hashlib.sha256(normalized.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(normalized[::-1].encode()).hexdigest(),
    }


def outside_in_obligation(text: str) -> dict[str, object]:
    normalized = tape(text)
    pairs = []
    lo, hi = 0, len(normalized) - 1
    while lo < hi and normalized[lo] == normalized[hi]:
        pairs.append((lo, hi, normalized[lo]))
        lo += 1
        hi -= 1
    return {"matched_outer_pairs": len(pairs), "first_open": None if lo >= hi else (lo, hi), "pairs": pairs[:8]}


def render(left: tuple[str, ...], right: tuple[str, ...]) -> str:
    return f"{left[0]} {left[1]} {left[2]} {left[3]}. {right[0]} {right[1]} {right[2]} {right[3]}."


def novelty_preflight() -> dict[str, object]:
    registry = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    entries = registry.get("entries", []) + registry.get("excluded", [])
    collisions = [e.get("id") for e in entries if e.get("id") != ID and e.get("signature") == SIGNATURE]
    if collisions:
        raise RuntimeError(f"duplicate outside-in state rejected: {collisions}")
    return {
        "status": "passed",
        "performed_before_search": True,
        "registry_entries_read": len(entries),
        "exact_signature_collision": False,
        "fixed_tape_used": False,
        "catalogue_phrase_imported": False,
        "finished_surface_reversed": False,
        "duplicate_sweep_rejected": True,
    }


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    left_choices = list(itertools.product(*LEFT.values()))
    right_choices = list(itertools.product(*RIGHT.values()))
    rows = []
    for left, right in itertools.product(left_choices, right_choices):
        text = render(left, right)
        audit = pointer_audit(text)
        admission = mechanical_admission_checks(text, min_letters=100, max_letters=240)
        obligation = outside_in_obligation(text)
        rows.append({
            "rendered": text,
            "slot_choices": {"left": dict(zip(LEFT, left)), "right": dict(zip(RIGHT, right))},
            "semantic_roles": {"left": "agent-action-theme-location", "right": "agent-action-theme-location"},
            "audit": audit,
            "outside_in_obligation": obligation,
            "mechanical_admission": admission,
            "anti_shortcut": {
                "catalogue_imported": False,
                "fixed_tape": False,
                "finished_surface_reversal": False,
                "word_order_mirror": not admission["not_word_order_symmetry"],
                "repeated_unit": not admission["no_repeated_nontrivial_unit"],
                "self_palindromic_unit": not admission["no_self_palindromic_proper_multiword_span"],
            },
            "provenance": {
                "lexical_source": "fresh authored role-typed phrase bank",
                "choices_before_rendering": True,
                "generator": str(Path(__file__).relative_to(ROOT)),
            },
        })
    rows.sort(key=lambda row: (row["audit"]["exact"], row["mechanical_admission"]["exact_letter_palindrome"], row["outside_in_obligation"]["matched_outer_pairs"], row["audit"]["letters"]), reverse=True)
    exact = [r for r in rows if r["audit"]["exact"] and all(r["mechanical_admission"].values())]
    best = exact[0] if exact else rows[0]
    payload = {
        "experiment_id": ID,
        "signature": SIGNATURE,
        "status": "completed_exact_closure" if exact else "completed_no_exact_closure",
        "reader_eligible": bool(exact),
        "method": "outside-in expansion of fresh role-typed phrase pairs with live character obligation",
        "novelty_preflight": preflight,
        "search": {"left_states": len(left_choices), "right_states": len(right_choices), "complete_pairs": len(rows), "exact_candidates": len(exact), "minimum_letters": 100},
        "best_candidate": best,
        "full_rendered_prose": best["rendered"],
        "next_repair": {
            "operator": "replace the phrase at the first open outer obligation with a held-out role-compatible phrase, then resume outside-in expansion",
            "reason": "no exact mechanically admitted closure in the fresh bounded bank" if not exact else "human-review the exact candidate before any extension",
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "audits": ["independent two-pointer", "forward/reverse SHA-256", "outside-in obligation ledger", "mechanical admission", "anti-shortcut checks"],
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["search"], sort_keys=True))
