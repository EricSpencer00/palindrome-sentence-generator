"""Fast lexical-hinge repair for exact English palindrome material.

The operator starts with independently authored, complete clauses.  It freezes
their letters, reverses that tape, and searches a fixed common-word lexicon for
new word boundaries on the other side.  A candidate is therefore a concrete
character-level repair, not a word-order mirror and not a relexicalized
catalogue entry. Exactness is mechanical, while sentencehood and meaning still
require a reader screen; failed surfaces are diagnostic records, never leads.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable

from wordfreq import zipf_frequency


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
MIN_SOURCE_LETTERS = 12
MIN_OUTPUT_LETTERS = 30
MAX_OUTPUT_LETTERS = 120
MAX_RIGHT_WORDS = 10

# These are fresh proposal material, kept in the experiment itself so the
# source can be audited without trusting a generated result file.
SOURCE_CLAUSES = (
    "a dog was in it today",
    "a dog was in it now",
    "i met a man",
    "the quiet nurse reads one warm letter",
    "our kind teacher writes one clear note",
    "the sailor saw a red boat",
    "we carried fresh water home",
)


def normalize(text: str) -> str:
    """Return the case- and punctuation-insensitive letter tape."""
    return "".join(ch for ch in text.lower() if "a" <= ch <= "z")


def words(text: str) -> tuple[str, ...]:
    return tuple(WORD_RE.findall(text.lower()))


def _common_lexicon(min_zipf: float = 3.0) -> set[str]:
    """Build a reproducible, frequency-limited lexicon from the local list."""
    path = ROOT / "data" / "lexicon.txt"
    return {
        word.strip()
        for word in path.read_text().splitlines()
        if re.fullmatch(r"[a-z]+", word.strip())
        and (len(word.strip()) >= 2 or word.strip() in {"a", "i"})
        and zipf_frequency(word.strip(), "en") >= min_zipf
    }


def _trie(vocabulary: Iterable[str]) -> dict:
    root: dict = {}
    for word in vocabulary:
        node = root
        for char in word:
            node = node.setdefault(char, {})
        node.setdefault("", True)
    return root


def segment_tape(tape: str, vocabulary: set[str], *, limit: int = 64) -> list[str]:
    """Enumerate lexical word breaks for one immutable tape."""
    if not tape.isalpha():
        return []
    trie = _trie(vocabulary)
    memo: dict[tuple[int, int], list[tuple[str, ...]]] = {}

    def visit(position: int, remaining_words: int) -> list[tuple[str, ...]]:
        key = (position, remaining_words)
        if key in memo:
            return memo[key]
        if position == len(tape):
            return [()] if remaining_words >= 3 else []
        if remaining_words >= MAX_RIGHT_WORDS:
            return []
        result: list[tuple[str, ...]] = []
        node = trie
        for end in range(position, len(tape)):
            node = node.get(tape[end])
            if node is None:
                break
            if "" not in node:
                continue
            for tail in visit(end + 1, remaining_words + 1):
                result.append((tape[position:end + 1],) + tail)
                if len(result) >= limit:
                    memo[key] = result
                    return result
        memo[key] = result
        return result

    return sorted({" ".join(row) for row in visit(0, 0)})[:limit]


def _word_order_mirror(units: tuple[str, ...]) -> bool:
    """Detect the forbidden trivial sequence mirror, independently."""
    half = len(units) // 2
    if len(units) % 2 == 0 and units[:half] == tuple(reversed(units[half:])):
        return True
    return False


def _catalogue() -> set[str]:
    path = ROOT / "data" / "known_palindromes.json"
    return set(json.loads(path.read_text()))


def _right_has_sentence_shape(right: str, vocabulary: set[str]) -> bool:
    """Conservative structural gate, never presented as a readability judge."""
    units = words(right)
    if not units or any(unit not in vocabulary for unit in units):
        return False
    # A tiny, explicit clause inventory is used only to detect obvious
    # fragments.  It is not a semantic classifier.
    subjects = {"a", "an", "the", "this", "that", "my", "our", "i", "we", "you", "he", "she", "they"}
    verbs = {
        "am", "are", "be", "been", "bring", "brought", "carry", "carried", "find", "found",
        "gave", "give", "go", "had", "has", "have", "held", "is", "keep", "kept", "left",
        "like", "lives", "made", "make", "met", "open", "opened", "read", "reads", "ran",
        "saw", "see", "sent", "sit", "sat", "takes", "took", "walk", "was", "were", "write", "writes",
    }
    return units[0] in subjects and bool(set(units) & verbs) and len(units) >= 3


def audit_candidate(left: str, right: str, catalogue: set[str], vocabulary: set[str]) -> dict[str, object]:
    """Recompute every hard property without trusting the constructor."""
    rendered = f"{left} {right}"
    tape = normalize(rendered)
    units = words(rendered)
    from llm_palindrome.admission import mechanical_admission_checks
    shared = mechanical_admission_checks(
        rendered, local_catalogue=catalogue, min_letters=MIN_OUTPUT_LETTERS,
        max_letters=MAX_OUTPUT_LETTERS,
    )
    checks = shared | {
        "minimum_length": len(tape) >= MIN_OUTPUT_LETTERS,
        "maximum_length": len(tape) <= MAX_OUTPUT_LETTERS,
        "all_words_in_lexicon": all(word in vocabulary for word in units),
        "not_word_order_mirror": not _word_order_mirror(units),
        "not_catalogued": shared["local_catalogue_absent"],
        "right_sentence_shape": _right_has_sentence_shape(right, vocabulary),
    }
    return {
        "left": left,
        "right": right,
        "rendered": rendered,
        "normalized": tape,
        "letters": len(tape),
        "checks": checks,
        "rejection_codes": [name for name, passed in checks.items() if not passed],
        "independent_exact_audit": tape == tape[::-1],
    }


def run(*, min_zipf: float = 3.0, limit_per_source: int = 8) -> dict[str, object]:
    """Run the frozen lexical-hinge repair and return all rendered surfaces."""
    vocabulary = _common_lexicon(min_zipf)
    catalogue = _catalogue()
    records: list[dict[str, object]] = []
    source_records: list[dict[str, object]] = []
    for source_id, left in enumerate(SOURCE_CLAUSES, 1):
        source_tape = normalize(left)
        source_row = {
            "source_id": f"S{source_id:02d}",
            "left": left,
            "source_normalized": source_tape,
            "source_letters": len(source_tape),
            "source_is_complete_clause": left in SOURCE_CLAUSES[:3],
        }
        source_records.append(source_row)
        # The hinge is the whole source clause: all source letters are frozen;
        # only the target's boundaries may change.
        right_tape = source_tape[::-1]
        lattice = segment_tape(right_tape, vocabulary, limit=limit_per_source)
        if not lattice:
            records.append({
                **source_row,
                "right": None,
                "rendered": None,
                "normalized": None,
                "letters": 0,
                "lattice_size": 0,
                "rejection_codes": ["no_exact_lexical_repair"],
                "independent_exact_audit": False,
            })
            continue
        for right in lattice:
            row = audit_candidate(left, right, catalogue, vocabulary)
            row |= {
                "source_id": source_row["source_id"],
                "source_normalized": source_tape,
                "reversed_source_tape": right_tape,
                "lattice_size": len(lattice),
                "provenance": "fresh source clause -> immutable reversed tape -> dictionary resegmentation",
            }
            records.append(row)
    admitted = [row for row in records if not row.get("rejection_codes")]
    vocabulary_hash = hashlib.sha256("\n".join(sorted(vocabulary)).encode()).hexdigest()
    return {
        "status": "complete_fast_luna_lexical_hinge_repair",
        "operator": "freeze a complete source clause, reverse its letters, and resegment the target tape",
        "config": {
            "min_zipf": min_zipf,
            "limit_per_source": limit_per_source,
            "min_output_letters": MIN_OUTPUT_LETTERS,
            "max_output_letters": MAX_OUTPUT_LETTERS,
        },
        "source_records": source_records,
        "records": records,
        "mechanically_admitted": admitted,
        "vocabulary_size": len(vocabulary),
        "vocabulary_sha256": vocabulary_hash,
        "catalogue_sha256": hashlib.sha256((ROOT / "data" / "known_palindromes.json").read_bytes()).hexdigest(),
        "reader_gate": (
            "Any future survivor must be shown as intact prose beside matched natural controls, "
            "with randomized blinded order and independent ratings of grammar, recoverable intent, "
            "and coherence. Automatic checks here make no readability claim."
        ),
        "next_reader_facing_test": {
            "design": "blind pairwise screen",
            "candidate_display": "one candidate with normal capitalization and terminal punctuation",
            "controls": ["ordinary prose of matched length", "word-shuffled control"],
            "ratings": ["grammatical", "recoverable intent", "coherent thought"],
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = run()
    if args.out:
        if args.out.exists():
            parser.error("refusing to overwrite an existing output")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out) if args.out else None,
        "records": len(result["records"]),
        "mechanically_admitted": len(result["mechanically_admitted"]),
        "candidates": [row["rendered"] for row in result["records"] if row.get("rendered")],
        "normalized": [row["normalized"] for row in result["records"] if row.get("rendered")],
        "lengths": [row["letters"] for row in result["records"] if row.get("rendered")],
    }, indent=2))
