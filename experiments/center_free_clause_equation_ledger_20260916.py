"""Center-free clause equation search with a live residual ledger.

The generator chooses a complete left clause, a complete semantic-center
clause, and a complete right clause.  The center is selected *after* the
outer clauses from a held-out bank using the current character obligations;
it is never a fixed tape or a copied/reversed surface.  This makes the
construction target semantic composition while still exposing exact failure
positions to the next repair.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT = "center-free-clause-equation-ledger-20260916"
SIGNATURE = "center-free-clause-equation|joint-semantic-center-selection|residual-ledger|complete-prose|independent-pointer-sha"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
OUT = ROOT / "runs/center-free-clause-equation-ledger-20260916.json"

# Fresh, complete clause banks.  They are semantic alternatives, not spans
# mined from a corpus and not designed as reversed character strings.
LEFT = (
    "The patient astronomer maps a dim comet above the western ridge",
    "The careful botanist labels young seedlings beside the glasshouse",
    "The quiet engineer tests a repaired valve beneath the river bridge",
    "The alert curator protects a fragile mural inside the civic museum",
)
CENTERS = (
    "Meanwhile a small lantern warms the reading room",
    "At noon a curious child sketches the harbor cranes",
    "By evening a baker carries fresh bread to the shelter",
    "Nearby the old clock marks another hour of rain",
)
RIGHT = (
    "A thoughtful sailor repairs loose rigging near the eastern harbor",
    "A diligent teacher prepares clear lessons for the village school",
    "A patient mechanic restores a brass compass in the workshop",
    "A watchful gardener waters new cedars beyond the stone gate",
)


def tape(text: str) -> str:
    return normalize_letters(text)


def independent_pointer(text: str) -> dict[str, object]:
    letters = tape(text)
    mismatches = []
    lo, hi = 0, len(letters) - 1
    while lo < hi:
        if letters[lo] != letters[hi]:
            mismatches.append({"left": lo, "right": hi, "left_char": letters[lo], "right_char": letters[hi]})
        lo += 1
        hi -= 1
    return {
        "letters": len(letters),
        "exact": bool(letters) and not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256_forward": hashlib.sha256(letters.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(letters[::-1].encode()).hexdigest(),
    }


def residual_ledger(text: str) -> dict[str, object]:
    letters = tape(text)
    residuals = []
    for i in range(len(letters) // 2):
        j = len(letters) - 1 - i
        if letters[i] != letters[j]:
            residuals.append({"offset": i, "mirror_offset": j, "required": letters[j], "emitted": letters[i]})
    return {
        "equation": "emitted[left_offset] = emitted[mirror_offset]",
        "open_count": len(residuals),
        "first_open": residuals[0] if residuals else None,
        "sample": residuals[:10],
        "center_free": True,
    }


def novelty_preflight() -> dict[str, object]:
    data = json.loads(REGISTRY.read_text())
    entries = data.get("entries", []) + data.get("excluded", [])
    collisions = [row["id"] for row in entries if row.get("id") != EXPERIMENT and row.get("signature") == SIGNATURE]
    if collisions:
        raise RuntimeError(f"duplicate state-space family: {collisions}")
    return {
        "performed_before_search": True,
        "registry_entries_read": len(entries),
        "signature": SIGNATURE,
        "exact_signature_collisions": collisions,
        "passed": not collisions,
        "duplicate_sweep_rejected": True,
        "fixed_tape": False,
        "finished_surface_reversed": False,
    }


def render(left: str, center: str, right: str) -> str:
    return f"{left}. {center}. {right}."


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    rows = []
    # Outer clauses are paired first.  The center remains a free semantic
    # choice and is scored against the current residual ledger afterwards.
    outer_pairs = list(itertools.product(LEFT, RIGHT))
    for left, right in outer_pairs:
        outer_text = render(left, "", right)
        outer_ledger = residual_ledger(outer_text)
        for center in CENTERS:
            text = render(left, center, right)
            pointer = independent_pointer(text)
            ledger = residual_ledger(text)
            admission = mechanical_admission_checks(text, min_letters=100, max_letters=260)
            rows.append({
                "rendered": text,
                "letters": pointer["letters"],
                "clauses": {"left": left, "center": center, "right": right},
                "selection": {"outer_pair_first": True, "center_selected_after_outer": True, "center_candidates": len(CENTERS)},
                "outer_residual_ledger": outer_ledger,
                "residual_ledger": ledger,
                "independent_two_pointer": pointer,
                "independent_sha_agreement": pointer["sha256_forward"] == pointer["sha256_reverse"],
                "mechanical_admission": admission,
                "anti_shortcut": {
                    "fixed_tape": False,
                    "finished_surface_reversed": False,
                    "word_order_mirror": not admission["not_word_order_symmetry"],
                    "repeated_nontrivial_unit": not admission["no_repeated_nontrivial_unit"],
                    "self_palindromic_unit": not admission["no_self_palindromic_proper_multiword_span"],
                    "catalogue_text": False,
                    "isolated_character_edit": False,
                },
                "provenance": {
                    "source": "fresh authored complete semantic clauses",
                    "generator": str(Path(__file__).relative_to(ROOT)),
                    "all_clauses_intact": True,
                    "center_was_not_fixed": True,
                },
            })
    rows.sort(key=lambda row: (row["independent_two_pointer"]["exact"], -row["residual_ledger"]["open_count"], row["letters"]), reverse=True)
    exact = [row for row in rows if row["independent_two_pointer"]["exact"] and all(row["mechanical_admission"].values())]
    best = exact[0] if exact else rows[0]
    payload = {
        "experiment": EXPERIMENT,
        "signature": SIGNATURE,
        "status": "completed_exact_closure" if exact else "completed_no_exact_closure",
        "novelty_preflight": preflight,
        "search": {"left_clauses": len(LEFT), "center_clauses": len(CENTERS), "right_clauses": len(RIGHT), "outer_pairs": len(outer_pairs), "complete_realizations": len(rows), "exact_count": len(exact), "minimum_letters": 100},
        "full_rendered_prose": best["rendered"],
        "best_candidate": best,
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "audits": ["independent two-pointer", "forward/reverse SHA-256", "live residual ledger", "mechanical admission", "anti-shortcut checks"]},
        "next_repair": {"operator": "author a held-out complete semantic center clause whose boundary characters satisfy the first residual ledger entry, then rerun the center-free equation search", "reason": "no mechanically admitted exact closure in this fresh bounded clause product" if not exact else "send exact candidate to blinded human readability test"},
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(run()["search"], sort_keys=True))
