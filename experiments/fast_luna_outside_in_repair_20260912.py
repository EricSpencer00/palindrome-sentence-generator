"""Fast outside-in repair experiment for long, literal English palindromes.

This deliberately does not enumerate mirrored lexical slots.  It starts with
complete, ordinary sentences, finds the first unequal character pair from the
outside in, and proposes a reversible same-POS word edit.  Every proposed
surface is then audited independently.  An exact short phrase is retained as
evidence of the boundary problem, but is not admitted as a result.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks

# A small, inspectable grammar inventory is used only to preserve a sentence's
# coarse shape while editing; it is not a Cartesian slot generator.
POS = {
    "DET": {"the", "this", "that", "your", "our", "one"},
    "ADJ": {"quiet", "patient", "warm", "clear", "calm", "small", "kind", "old"},
    "NOUN": {"nurse", "baker", "loaf", "letter", "teacher", "note", "sailor", "map", "reader", "garden"},
    "VERB": {"reads", "carries", "writes", "guides", "keeps", "visits", "marks"},
}
REPLACEMENTS = {
    "ADJ": ("quiet", "patient", "warm", "clear", "calm", "small", "kind", "old"),
    "NOUN": ("nurse", "baker", "loaf", "letter", "teacher", "note", "sailor", "map", "reader", "garden"),
    "VERB": ("reads", "carries", "writes", "guides", "keeps", "visits", "marks"),
}
SEEDS = (
    "The quiet nurse reads one warm letter.",
    "Your patient baker carries one warm loaf.",
    "Our kind teacher writes one clear note.",
    "This calm sailor carries one small map.",
)
# Exact, non-repeated, non-self-palindromic evidence.  It is intentionally a
# fragment and under the long-sentence threshold, so it cannot pass the gate.
EXACT_NEAR_MISS = "Deep state led art trade let at speed."


def normalize(text: str) -> str:
    """ASCII letters only: punctuation and spacing cannot contribute letters."""
    return "".join(ch.lower() for ch in text if "a" <= ch.lower() <= "z")


def words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", text.lower()))


def mismatch_pairs(tape: str) -> tuple[tuple[int, int], ...]:
    return tuple((i, len(tape) - 1 - i) for i in range(len(tape) // 2) if tape[i] != tape[-1 - i])


def pos_of(word: str) -> str | None:
    return next((pos for pos, members in POS.items() if word in members), None)


def coarse_grammar(text: str) -> bool:
    """Require one complete SVO reading, not a fragment or word salad."""
    tagged = tuple(pos_of(word) for word in words(text))
    return tagged in {
        ("DET", "ADJ", "NOUN", "VERB", "DET", "ADJ", "NOUN"),
        ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"),
    } and text[:1].isupper() and text.rstrip().endswith(".")


def reverse_word_pairing(units: tuple[str, ...]) -> bool:
    """Reject word-order mirrors and spelling-reversed word pairs."""
    n = len(units)
    if n % 2 == 0 and units[: n // 2] == tuple(reversed(units[n // 2 :])):
        return True
    half = n // 2
    if n % 2 == 0 and all(a == b[::-1] for a, b in zip(units[:half], reversed(units[half:]))):
        return True
    return any(a != b and a == b[::-1] for i, a in enumerate(units) for b in units[i + 1 :])


def strict_validation(text: str, catalogue: set[str]) -> dict[str, object]:
    units = words(text)
    tape = normalize(text)
    punctuation_neutral = bool(re.fullmatch(r"[A-Za-z .,!?'-]+", text)) and normalize(text) == tape
    shared = mechanical_admission_checks(
        text, local_catalogue=catalogue, min_letters=40, max_letters=120
    )
    checks = shared | {
        "at_least_40_letters": len(tape) >= 40,
        "complete_english_grammar": coarse_grammar(text),
        "no_repeated_units": shared["distinct_words"] and shared["no_repeated_nontrivial_unit"],
        "no_self_palindromic_units": shared["no_self_palindromic_word"],
        "no_reverse_word_pairing": not reverse_word_pairing(units),
        "punctuation_does_not_change_letters": punctuation_neutral,
        "not_catalogue_material": shared["local_catalogue_absent"],
    }
    return {"rendered": text, "normalized": tape, "length": len(tape), "checks": checks,
            "admitted": all(checks.values())}


def repair_once(text: str) -> dict[str, object]:
    """Apply one reversible outside-in same-POS edit, if it lowers mismatch."""
    before = normalize(text)
    mismatches = mismatch_pairs(before)
    baseline = len(mismatches)
    units = list(words(text))
    # Locate the first unequal pair in the letter tape, then restrict edits to
    # the lexical item crossing either endpoint.  This is the outside-in
    # topology, rather than a score over arbitrary slot substitutions.
    seam = mismatches[0] if mismatches else None
    spans: list[tuple[int, int]] = []
    cursor = 0
    for unit in units:
        spans.append((cursor, cursor + len(unit)))
        cursor += len(unit)
    seam_words = ({index for index, (start, end) in enumerate(spans)
                   if seam and ((start <= seam[0] < end) or (start <= seam[1] < end))})
    best: tuple[int, str, int, str] | None = None
    for index, old in enumerate(units):
        if seam is not None and index not in seam_words:
            continue
        pos = pos_of(old)
        if pos not in REPLACEMENTS:
            continue
        for replacement in REPLACEMENTS[pos]:
            if replacement == old or replacement in units:
                continue
            trial_units = units[:]
            trial_units[index] = replacement
            trial = " ".join(trial_units).capitalize() + "."
            score = len(mismatch_pairs(normalize(trial)))
            if score < baseline and (best is None or score < best[0]):
                best = (score, trial, index, old + "->" + replacement)
    if best is None:
        return {"source": text, "rendered": None, "operator": "same_pos_outside_in_edit",
                "mismatches_before": baseline, "mismatches_after": None}
    score, rendered, index, edit = best
    return {"source": text, "rendered": rendered, "operator": "same_pos_outside_in_edit",
            "edited_word_index": index, "edit": edit, "mismatches_before": baseline,
            "mismatches_after": score}


def next_repair_operator(text: str) -> dict[str, object]:
    """Concrete next repair: jointly edit two same-POS words around the first seam.

    The pair edit is still reversible and grammar-preserving, but can escape a
    one-word local minimum.  It is intentionally bounded to keep this run fast.
    """
    units = list(words(text))
    mismatches = mismatch_pairs(normalize(text))
    baseline = len(mismatches)
    seam = mismatches[0] if mismatches else None
    spans: list[tuple[int, int]] = []
    cursor = 0
    for unit in units:
        spans.append((cursor, cursor + len(unit)))
        cursor += len(unit)
    seam_words = ({index for index, (start, end) in enumerate(spans)
                   if seam and ((start <= seam[0] < end) or (start <= seam[1] < end))})
    candidates = [(i, pos_of(word)) for i, word in enumerate(units)
                  if pos_of(word) in REPLACEMENTS and (seam is None or i in seam_words)]
    best: tuple[int, str, tuple[int, int]] | None = None
    for ai, (i, pos_i) in enumerate(candidates):
        for j, pos_j in candidates[ai + 1 :]:
            for left in REPLACEMENTS[pos_i]:
                for right in REPLACEMENTS[pos_j]:
                    if left == units[i] or right == units[j] or left == right:
                        continue
                    trial_units = units[:]
                    trial_units[i], trial_units[j] = left, right
                    if len(set(trial_units)) != len(trial_units):
                        continue
                    trial = " ".join(trial_units).capitalize() + "."
                    score = len(mismatch_pairs(normalize(trial)))
                    if score < baseline and (best is None or score < best[0]):
                        best = (score, trial, (i, j))
    return {"operator": "bounded_two_word_same_pos_repair", "source": text,
            "mismatches_before": baseline,
            "rendered": best[1] if best else None,
            "mismatches_after": best[0] if best else None,
            "edited_word_indices": best[2] if best else None}


def run() -> dict[str, object]:
    known_path = ROOT / "data" / "known_palindromes.json"
    catalogue = set(json.loads(known_path.read_text()))
    rendered = [EXACT_NEAR_MISS, *SEEDS]
    repairs = [repair_once(seed) for seed in SEEDS]
    for repair in repairs:
        if repair["rendered"]:
            rendered.append(str(repair["rendered"]))
    audits = [strict_validation(text, catalogue) for text in rendered]
    next_repairs = [next_repair_operator(seed) for seed in SEEDS]
    return {
        "status": "failed_long_readable_search_with_repair_evidence",
        "method": "outside_in_first_mismatch_then_reversible_same_POS_edit",
        "provenance": {"seed_sentences": SEEDS, "exact_near_miss": EXACT_NEAR_MISS,
                       "lexicon": "inline closed POS inventory; no lexical slot enumeration",
                       "catalogue_path": str(known_path)},
        "candidates": audits,
        "repair_trace": repairs,
        "next_repair_trace": next_repairs,
        "next_repair_operator": "bounded_two_word_same_pos_repair around the first outside-in seam, then reparse the full sentence",
        "reader_facing_next_test": "Blind readers should rate any future 40+ letter strict survivors for completeness, ordinary meaning, and ease of one-pass reading; this run has no admitted survivor.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run()
    output = json.dumps(result, indent=2) + "\n"
    if args.out:
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    print(output, end="")
