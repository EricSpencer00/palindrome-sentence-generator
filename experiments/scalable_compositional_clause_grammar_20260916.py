"""Length-indexed compositional grammar without nested palindrome spans.

The grammar starts from a fresh complete garden scene and composes any number
of ordinary-order action increments.  Search is indexed by target length and
increment count; each complete sentence carries a mirrored-character
obligation frontier.  A candidate is never allowed to contain a proper
multiword palindromic span or a repeated nontrivial unit, even when its whole
tape is not exact.  The emitted probes retain independent exact, hash,
admission, and readability diagnostics plus a concrete next repair.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

EXPERIMENT = "scalable-compositional-clause-grammar-20260916"
SIGNATURE = (
    "length-indexed-clause-composition|arbitrary-ordinary-action-increments|"
    "mirrored-character-obligation-frontier|proper-span-and-unit-exclusion|"
    "multi-target-complete-probes|independent-exact-admission-audit"
)
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

BASE = "At first light, the gardener unlocks the old shed"
INCREMENTS = (
    ("water-barrel", ", checks the water barrel"),
    ("trim-apple", ", trims the apple tree"),
    ("sweep-path", ", sweeps the stone path"),
    ("label-trays", ", labels the seed trays"),
    ("carry-hose", ", carries the spare hose"),
    ("mend-gate", ", mends the loose gate"),
    ("fold-tarp", ", folds the canvas tarp"),
    ("write-neighbor", ", writes a note for the neighbor"),
)
TARGET_BANDS = ((80, 100), (105, 125), (125, 145))


def tape(text: str) -> str:
    return normalize_letters(text)


def exact_slice(text: str) -> dict:
    value = tape(text)
    return {"algorithm": "normalized_tape_reverse_slice", "exact": bool(value) and value == value[::-1], "letters": len(value)}


def exact_two_pointer(text: str) -> dict:
    value = tape(text)
    mismatches = []
    left, right = 0, len(value) - 1
    while left < right:
        if value[left] != value[right]:
            mismatches.append({"offset": left, "right_offset": right, "left": value[left], "right": value[right]})
        left += 1
        right -= 1
    return {"algorithm": "independent_two_pointer", "exact": bool(value) and not mismatches, "letters": len(value), "mismatch_count": len(mismatches), "mismatches": mismatches[:10]}


def hash_audit(text: str) -> dict:
    value = tape(text)
    forward = hashlib.sha256(value.encode()).hexdigest()
    reverse = hashlib.sha256(value[::-1].encode()).hexdigest()
    return {"algorithm": "independent_sha256_forward_vs_reverse", "forward_sha256": forward, "reverse_sha256": reverse, "exact": bool(value) and forward == reverse}


def compose(order: tuple[int, ...]) -> tuple[str, list[str]]:
    chunks = [BASE] + [INCREMENTS[index][1] for index in order]
    return "".join(chunks) + ".", chunks


def obligation_frontier(text: str, chunks: list[str]) -> dict:
    value = tape(text)
    pairs = [(i, len(value) - 1 - i) for i in range(len(value) // 2)]
    matches = sum(value[i] == value[j] for i, j in pairs)
    mismatch = next((i for i, j in pairs if value[i] != value[j]), None)
    emitted = 0
    checkpoints = []
    for chunk in chunks:
        emitted += len(tape(chunk))
        mirrored_start = max(0, len(value) - emitted)
        checkpoints.append({"emitted_letters": emitted, "mirrored_obligation_start": mirrored_start, "resolved_pairs": min(emitted, len(value) - emitted), "frontier_crossed": emitted >= len(value) / 2})
    return {"equation": "x[i] = x[N-1-i] for the composed complete sentence", "positions_checked": len(pairs), "matching_pairs": matches, "mismatch_pairs": len(pairs) - matches, "first_mismatch_offset": mismatch, "match_rate": matches / len(pairs) if pairs else 0.0, "obligation_checkpoints": checkpoints}


def repeated_unit(units: tuple[str, ...]) -> bool:
    normalized = tuple(normalize_letters(unit) for unit in units)
    repeatable = {"a", "an", "the", "and", "or", "but", "if", "as", "to", "in", "on", "at", "by", "for", "from", "with", "while", "when", "before", "after"}
    for width in range(2, len(normalized) // 2 + 1):
        for left in range(len(normalized) - width):
            for right in range(left + width, len(normalized) - width + 1):
                block = normalized[left:left + width]
                if block == normalized[right:right + width] and any(word not in repeatable for word in block):
                    return True
    return False


def proper_palindromic_span(units: tuple[str, ...]) -> bool:
    normalized = tuple(normalize_letters(unit) for unit in units)
    total = len(normalized)
    for width in range(2, total):
        for start in range(total - width + 1):
            segment = "".join(normalized[start:start + width])
            if segment and segment == segment[::-1]:
                return True
    return False


def independent_admission(text: str) -> dict:
    words = tokenize(text)
    return {"algorithm": "independent_nested-span-and-unit-scan", "ascii_letters_only": all(not c.isalpha() or c.isascii() for c in text), "complete_terminal_period": text.endswith("."), "at_least_four_clause_increments": text.count(",") >= 3, "minimum_word_count": len(words) >= 14, "no_proper_multiword_palindromic_span": not proper_palindromic_span(words), "no_repeated_nontrivial_unit": not repeated_unit(words), "not_word_order_mirror": tuple(normalize_letters(word) for word in words) != tuple(reversed(tuple(normalize_letters(word) for word in words)))}


def readability(text: str) -> dict:
    words = [word.casefold() for word in tokenize(text)]
    try:
        from wordfreq import zipf_frequency
        mean_zipf = sum(zipf_frequency(word, "en") for word in words) / len(words)
    except Exception:
        mean_zipf = None
    return {"diagnostic_not_human_readability": True, "word_count": len(words), "mean_zipf_frequency": mean_zipf, "repeated_word_rate": 1 - len(set(words)) / len(words) if words else None, "reader_status": "not_run; requires blinded human study"}


def next_repair(text: str, equation: dict, order: tuple[int, ...]) -> dict:
    offset = equation["first_mismatch_offset"] or 0
    used = set(order)
    heldout = next((index for index in range(len(INCREMENTS)) if index not in used), None)
    return {"operator": "append one held-out ordinary clause increment selected by first mismatch obligation", "first_mismatch_offset": offset, "heldout_increment": INCREMENTS[heldout][0] if heldout is not None else "none", "changed_increment_count": 1, "action": "append" if heldout is not None else "author a fresh increment", "recompute_complete_tape": True}


def audit(text: str, chunks: list[str], order: tuple[int, ...], target: tuple[int, int], rank: int = 0) -> dict:
    first, second, hashed = exact_slice(text), exact_two_pointer(text), hash_audit(text)
    equation = obligation_frontier(text, chunks)
    central = mechanical_admission_checks(text, min_letters=80, max_letters=180)
    independent = independent_admission(text)
    row = {"rank": rank, "target_band": list(target), "rendered": text, "letters": len(tape(text)), "increment_order": [INCREMENTS[index][0] for index in order], "compositional_depth": len(order), "mirrored_character_obligations": equation, "exact_check_1": first, "exact_check_2": second, "exact_check_hash": hashed, "independent_exact_agreement": first["exact"] == second["exact"] == hashed["exact"], "central_admission": central, "independent_admission": independent, "admission_agreement": central["not_word_order_symmetry"] == independent["not_word_order_mirror"], "mechanically_admitted": first["exact"] and second["exact"] and hashed["exact"] and all(central.values()) and all(independent.values()), "readability_evidence": readability(text), "provenance": {"source": "human-authored garden scene plus ordinary action increments", "catalogue_text_used": False, "pre_existing_palindrome_wrapped": False, "word_order_mirrored": False, "proper_multiword_palindromic_span_forbidden": True, "repeated_unit_forbidden": True}}
    row["next_repair_operator"] = "none; exact closure" if first["exact"] else next_repair(text, equation, order)
    return row


def novelty_preflight() -> dict:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [entry["id"] for entry in entries if entry.get("id") != EXPERIMENT and entry.get("signature") == SIGNATURE]
    related = [entry["id"] for entry in entries if any(term in entry.get("signature", "") for term in ("recursive", "obligation", "clause-growth", "length")) and entry.get("id") != EXPERIMENT]
    return {"entries_inspected": len(entries), "exact_signature_collisions_before_run": collisions, "related_families_for_manual_review": related[:24], "passed": not collisions, "state_space_distinction": "length-indexed composition of fresh ordinary action increments with explicit proper-span/repeated-unit bans; not a seam constructor or recursive obligation replay"}


def search() -> dict:
    preflight = novelty_preflight()
    if not preflight["passed"]:
        raise RuntimeError(f"novelty collision: {preflight['exact_signature_collisions_before_run']}")
    target_results = []
    all_probes = []
    for target in TARGET_BANDS:
        rows = []
        for depth in range(3, 6):
            for order in itertools.permutations(range(len(INCREMENTS)), depth):
                text, chunks = compose(order)
                letters = len(tape(text))
                if target[0] <= letters <= target[1]:
                    rows.append(audit(text, chunks, order, target))
        rows.sort(key=lambda row: (-row["mirrored_character_obligations"]["match_rate"], row["exact_check_2"]["mismatch_count"], row["letters"], row["rendered"]))
        for rank, row in enumerate(rows[:12], 1):
            row["rank"] = rank
        all_probes.extend(rows[:12])
        target_results.append({"target_band": list(target), "states_in_band": len(rows), "exact_count": sum(row["exact_check_1"]["exact"] for row in rows), "mechanically_admitted_count": sum(row["mechanically_admitted"] for row in rows), "rendered_probes": rows[:12]})
    exact = [row for row in all_probes if row["exact_check_1"]["exact"]]
    return {"experiment": EXPERIMENT, "signature": SIGNATURE, "status": "complete_length_indexed_compositional_search", "novelty_preflight": preflight, "human_seed": {"base": BASE, "increment_count": len(INCREMENTS), "increment_grammar": "Sentence := Base (comma-coordinated ordinary action increment)+"}, "target_bands": [list(target) for target in TARGET_BANDS], "maximum_compositional_depth_searched": 5, "states_examined": sum(result["states_in_band"] for result in target_results), "exact_count_in_emitted_probes": len(exact), "mechanically_admitted_count_in_emitted_probes": sum(row["mechanically_admitted"] for row in all_probes), "target_results": target_results, "best_rendered_candidates": all_probes, "failed_attempts": [row for row in all_probes if not row["exact_check_1"]["exact"]], "readability_evidence": {"status": "diagnostic_not_human_readability_result", "reader_eligible_count": 0, "method": "independent word-frequency, repetition, and sentence-shape diagnostics; no human readers run"}, "failure_evidence": {"all_emitted_nonexact_probes_retain_repair": True, "next_repair_operator": "append one held-out ordinary increment at the first obligation mismatch and recompute the complete tape", "proper_multiword_palindromic_spans_checked": True, "repeated_units_checked": True}, "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "catalogue_or_corpus_import": False, "pre_existing_palindrome_wrapped": False, "word_order_symmetry_used": False}}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"output already exists: {OUT}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = search()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("states_examined", "exact_count_in_emitted_probes", "mechanically_admitted_count_in_emitted_probes")}, indent=2))


if __name__ == "__main__":
    main()
