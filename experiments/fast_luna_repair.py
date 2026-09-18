"""Fast local repair for independently authored palindrome sentences.

The operator starts with ordinary sentence proposals and freezes every word
whose characters are outside the current mismatch window.  Only words touching
an unmatched pair may be replaced, and every replacement is selected from an
explicit lexical menu.  The left and right tape constraints are checked at the
same time; no reflected phrase, copied catalogue string, or punctuation can
create a result.

This is intentionally a constructive experiment rather than a readability
claim.  A mechanically surviving sentence is packaged for a later blinded
reader screen with an intact-prose control and a deterministic word-shuffle
control.  The default seeds are original ordinary sentences and are not read
from the palindrome catalogue.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.validator import is_palindrome, normalize
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.I)
SENTENCE_RE = re.compile(r"^[A-Za-z][A-Za-z ,;:'!?-]*[.!?]$")

# These are proposal material, not a source of palindromes.  They make the
# fast run reproducible when the local model is unavailable.
DEFAULT_SEEDS = (
    "Careful editors revise old drafts.",
    "Patient teachers guide young readers.",
    "Curious nurses examine small wounds.",
    "Quiet bakers warm fresh bread.",
    "Local artists repair broken frames.",
)

# The menu is deliberately small and transparent.  It is enough to exercise
# repair without silently turning this experiment into a corpus miner.
REPAIR_WORDS = frozenset(
    "a an and artists bakers careful children curious drafts editors examine "
    "fresh frames guide local nurses old patient repair readers revise small "
    "teachers the warm wounds young quiet broken bread artists careful".split()
)


@dataclass(frozen=True)
class RepairCandidate:
    """One mechanically exact local-repair result and its provenance."""

    text: str
    seed: str
    changed_slots: tuple[int, ...]
    mismatch_slots: tuple[int, ...]


def words(text: str) -> tuple[str, ...]:
    """Return lowercase word units while keeping punctuation out of the tape."""
    return tuple(WORD_RE.findall(text.lower()))


def word_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return normalized-letter spans for each displayed word."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for word in words(text):
        width = len(normalize(word))
        spans.append((cursor, cursor + width))
        cursor += width
    return tuple(spans)


def first_mismatch(tape: str) -> int | None:
    """Return the first left-to-right mismatch against the reversed tape."""
    for index, (left, right) in enumerate(zip(tape, reversed(tape))):
        if left != right:
            return index
    return None


def mismatch_positions(tape: str) -> tuple[int, ...]:
    """Return both character positions for every unequal mirrored pair."""
    positions: set[int] = set()
    for index in range(len(tape) // 2):
        opposite = len(tape) - 1 - index
        if tape[index] != tape[opposite]:
            positions.update((index, opposite))
    return tuple(sorted(positions))


def mismatch_word_slots(text: str) -> tuple[int, ...]:
    """Map mismatching tape characters back to the word slots they touch."""
    tape = normalize(text)
    bad = set(mismatch_positions(tape))
    slots = []
    for slot, (start, end) in enumerate(word_spans(text)):
        if bad.intersection(range(start, end)):
            slots.append(slot)
    return tuple(slots)


def word_order_mirror(units: Sequence[str]) -> bool:
    """Reject the old mirror-phrase construction at word boundaries."""
    units = tuple(units)
    for cut in range(1, len(units)):
        left, right = units[:cut], units[cut:]
        if len(left) == len(right) and tuple(word[::-1] for word in left[::-1]) == right:
            return True
    return False


def repeated_unit(units: Sequence[str]) -> bool:
    """Reject repeated contiguous units, including repeated blocks."""
    units = tuple(units)
    if len(units) != len(set(units)):
        return True
    for width in range(1, len(units) // 2 + 1):
        for start in range(len(units) - 2 * width + 1):
            if units[start:start + width] == units[start + width:start + 2 * width]:
                return True
    return False


def screen(text: str, *, vocabulary: set[str], known: set[str],
           min_letters: int = 30, max_letters: int = 90) -> dict[str, bool]:
    """Apply only mechanical gates; grammar/readability stays a reader gate."""
    tape = normalize(text)
    units = words(text)
    shared = mechanical_admission_checks(
        text, local_catalogue=known, min_letters=min_letters, max_letters=max_letters
    )
    return shared | {
        "sentence_form": bool(SENTENCE_RE.fullmatch(text.strip())),
        "punctuation_display_only": (
            bool(units) and normalize(text) == "".join(units)
            and not bool(re.search(r"[.!?]{2,}", text))
        ),
        "lexicon_words": bool(units) and all(unit in vocabulary for unit in units),
        "no_repeated_words": shared["distinct_words"],
        "no_self_palindromic_word_units": shared["no_self_palindromic_word"],
        "no_word_order_mirror": not word_order_mirror(units),
        "no_repeated_units": not repeated_unit(units),
        "novel_local_catalogue": tape not in known,
    }


def local_repair(seed: str, alternatives: Mapping[int, Sequence[str]], *,
                 max_mutations: int = 2, limit: int = 64) -> list[RepairCandidate]:
    """Repair a proposal by changing only words touching a mismatch.

    The seed's word boundaries remain fixed.  Alternatives for untouched slots
    are ignored, and a candidate is emitted only after independent whole-tape
    equality.  This is exhaustive for the supplied local menus (up to
    ``limit`` results), so a failed run still demonstrates a real repair
    attempt rather than a failed prompt-only strategy.
    """
    base = list(words(seed))
    if not base:
        return []
    mutable = mismatch_word_slots(seed)
    if not mutable:
        return []
    menus: list[tuple[str, ...]] = []
    for slot, original in enumerate(base):
        if slot not in mutable:
            menus.append((original,))
            continue
        choices = [original, *(str(value).lower() for value in alternatives.get(slot, ()))]
        menus.append(tuple(dict.fromkeys(value for value in choices if WORD_RE.fullmatch(value))))
    out: list[RepairCandidate] = []
    for selected in itertools.product(*menus):
        changed = tuple(index for index, (before, after) in enumerate(zip(base, selected))
                        if before != after)
        if not changed or len(changed) > max_mutations:
            continue
        # Punctuation is preserved only as a display suffix; it cannot repair
        # the tape.  Seeds are sentence-shaped and this keeps their casing sane.
        punctuation = "." if seed.rstrip().endswith(".") else "?" if seed.rstrip().endswith("?") else ""
        text = " ".join(selected).capitalize() + punctuation
        if is_palindrome(text):
            out.append(RepairCandidate(text, seed, changed, mutable))
            if len(out) >= limit:
                break
    return out


def reader_test_package(text: str) -> dict:
    """Prepare an intact candidate and matched shuffle without judging either."""
    units = list(words(text))
    # A stable non-palindromic control with the same words; no randomness is
    # needed for this development package, while a study can randomize order.
    shuffled = units[1::2] + units[::2] if len(units) > 2 else list(reversed(units))
    terminal = re.search(r"[.!?]$", text.rstrip())
    mark = terminal.group(0) if terminal else "."
    return {
        "status": "not_run",
        "candidate_intact": text,
        "matched_word_shuffle": " ".join(shuffled) + mark,
        "instructions": "Randomize intact and shuffle items; ask independent readers for grammar, coherence, and intent ratings.",
    }


def _known_catalogue() -> set[str]:
    path = ROOT / "data" / "known_palindromes.json"
    try:
        return set(json.loads(path.read_text()))
    except FileNotFoundError:
        return set()


def _default_alternatives(seeds: Sequence[str]) -> dict[int, tuple[str, ...]]:
    """Build a bounded ordinary-word menu for every slot in the frozen seeds."""
    count = max((len(words(seed)) for seed in seeds), default=0)
    menu = ("careful", "patient", "curious", "quiet", "local", "young",
            "old", "new", "fresh", "broken", "repair", "guide")
    return {slot: menu for slot in range(count)}


def run(*, seeds: Sequence[str] = DEFAULT_SEEDS, alternatives: Mapping[int, Sequence[str]] | None = None,
        vocabulary: set[str] | None = None, min_letters: int = 30) -> dict:
    """Run local repair and return complete provenance, including empty runs."""
    if alternatives is None:
        alternatives = _default_alternatives(seeds)
    vocabulary = set(vocabulary or REPAIR_WORDS)
    known = _known_catalogue()
    attempts: list[dict] = []
    survivors: list[dict] = []
    for seed in seeds:
        repairs = local_repair(seed, alternatives)
        seed_row = {"seed": seed, "mismatch_slots": list(mismatch_word_slots(seed)),
                    "mismatch_positions": list(mismatch_positions(normalize(seed))),
                    "first_mismatch": first_mismatch(normalize(seed)),
                    "mutable_slots": list(mismatch_word_slots(seed)),
                    "repair_menu_sizes": {
                        str(slot): len(alternatives.get(slot, ()))
                        for slot in mismatch_word_slots(seed)
                    },
                    "repairs": []}
        for candidate in repairs:
            checks = screen(candidate.text, vocabulary=vocabulary, known=known,
                            min_letters=min_letters)
            row = {"text": candidate.text, "seed": candidate.seed,
                   "changed_slots": list(candidate.changed_slots),
                   "mismatch_slots": list(candidate.mismatch_slots),
                   "letters": len(normalize(candidate.text)), "checks": checks,
                   "rejection_codes": [name for name, passed in checks.items() if not passed],
                   "reader_status": "unreviewed; mechanical exactness is not readability"}
            seed_row["repairs"].append(row)
            if not row["rejection_codes"]:
                row["reader_test"] = reader_test_package(candidate.text)
                survivors.append(row)
        attempts.append(seed_row)
    provenance = {
        "operator": "mismatch-local-bidirectional-word-repair",
        "seed_texts": list(seeds),
        "alternative_slots": {str(k): list(v) for k, v in alternatives.items()},
        "vocabulary_sha256": hashlib.sha256("\n".join(sorted(vocabulary)).encode()).hexdigest(),
        "known_catalogue_sha256": hashlib.sha256(
            json.dumps(sorted(known)).encode()).hexdigest(),
        "constraints": ["frozen word boundaries", "only mismatch-touching slots mutable",
                        "exact normalized tape equality", "no mirror phrase", "no repeats",
                        "no self-palindromic words", "local catalogue absence"],
    }
    return {
        "status": "complete_fast_luna_local_repair_run",
        "provenance": provenance,
        "attempts": attempts,
        "mechanically_surviving_candidates": survivors,
        "reader_gate": "No candidate is a readability claim; run the supplied intact-vs-shuffle blind screen before reporting readable prose.",
        "failure_repair_operator": "local_repair(seed, alternatives): exhaustive substitutions over mismatch-touching word slots with independent whole-tape verification",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-letters", type=int, default=30)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite an existing output")
    result = run(min_letters=args.min_letters)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "attempts": len(result["attempts"]),
                      "survivors": len(result["mechanically_surviving_candidates"])}, indent=2))


if __name__ == "__main__":
    main()
