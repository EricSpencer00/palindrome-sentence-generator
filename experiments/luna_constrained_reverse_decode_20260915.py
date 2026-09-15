"""Bounded reverse-tape lexical decoding outside the Brown/POS families.

Each fresh natural left clause supplies a fixed target tape.  The decoder
enumerates only dictionary-backed prefixes of that reversed tape, retaining a
small frequency-ranked frontier at each offset.  It never guesses an
incompatible right-side letter and never treats a partial parse as a
candidate.  Exact closures still pass the shared mechanical gate and remain
unreviewed until a blinded reader test.
"""
from __future__ import annotations

import glob
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import sys

from wordfreq import zipf_frequency

sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.lexicon import is_real_word, load_lexicon

ROOT = Path(__file__).parents[1]
LEFTS = (
    "Careful makers restore old radios.",
    "Bright students solve hard puzzles.",
    "Patient nurses record each dosage.",
    "Quiet artists frame winter scenes.",
)
MAX_WORDS = 8
MAX_WORD_LENGTH = 14
FRONTIER_CAP = 96


def repository_tapes(output: Path) -> tuple[set[str], int]:
    """Fingerprint all JSON strings before writing this run's output."""
    keys: set[str] = set()
    files = 0
    output = output.resolve()
    paths = glob.glob(str(ROOT / "data" / "*.json"))
    paths += glob.glob(str(ROOT / "runs" / "**" / "*.json*"), recursive=True)
    for raw in paths:
        path = Path(raw)
        if path.resolve() == output:
            continue
        try:
            value = json.loads(path.read_text())
        except Exception:
            continue
        files += 1

        def walk(item):
            if isinstance(item, str):
                try:
                    keys.add(normalize_letters(item))
                except ValueError:
                    pass
            elif isinstance(item, dict):
                for child in item.values():
                    walk(child)
            elif isinstance(item, list):
                for child in item:
                    walk(child)

        walk(value)
    return keys, files


def decode(target: str, vocabulary: frozenset[str], cap: int = FRONTIER_CAP):
    """Return frequency-ranked complete lexical segmentations of ``target``."""
    words_by_first: dict[str, tuple[str, ...]] = {}
    for word in vocabulary:
        if not word.isalpha() or not 2 <= len(word) <= MAX_WORD_LENGTH:
            continue
        words_by_first.setdefault(word[0], tuple())
    for first in tuple(words_by_first):
        words_by_first[first] = tuple(
            sorted(
                (word for word in vocabulary if word.isalpha() and 2 <= len(word) <= MAX_WORD_LENGTH and word.startswith(first)),
                key=lambda word: (-zipf_frequency(word, "en"), -len(word), word),
            )
        )

    @lru_cache(maxsize=None)
    def rec(offset: int, words_left: int):
        if offset == len(target):
            return ((),)
        if words_left == 0:
            return ()
        out = []
        for word in words_by_first.get(target[offset], ()):
            end = offset + len(word)
            if end > len(target) or not target.startswith(word, offset):
                continue
            for tail in rec(end, words_left - 1):
                out.append((word,) + tail)
                if len(out) >= cap * 2:
                    break
            if len(out) >= cap * 2:
                break
        out.sort(key=lambda row: (-sum(zipf_frequency(word, "en") for word in row), len(row), row))
        return tuple(out[:cap])

    complete = rec(0, MAX_WORDS)
    # Preserve the best constructive evidence even when no complete parse
    # exists: this bounded frontier is the exact reverse-compatible prefix
    # that the next operator must repair.
    frontier = [(0, (), 0.0)]
    best = frontier[0]
    for _ in range(MAX_WORDS):
        expanded = []
        for offset, words, score in frontier:
            if offset == len(target):
                continue
            for word in words_by_first.get(target[offset], ()):
                end = offset + len(word)
                if end > len(target) or not target.startswith(word, offset):
                    continue
                expanded.append((end, words + (word,), score + zipf_frequency(word, "en")))
        if not expanded:
            break
        expanded.sort(key=lambda row: (-row[0], -row[2], len(row[1]), row[1]))
        frontier = expanded[:cap]
        if frontier[0][:1] > best[:1] or (frontier[0][0] == best[0] and frontier[0][2] > best[2]):
            best = frontier[0]
    return complete, rec.cache_info(), {"consumed": best[0], "words": list(best[1]), "score": round(best[2], 3)}


def run(output: Path | None = None):
    output = output or ROOT / "runs" / "luna_constrained_reverse_decode_20260915.json"
    vocabulary = load_lexicon(str(ROOT / "data" / "lexicon.txt"))
    vocabulary = frozenset(word for word in vocabulary if is_real_word(word, vocabulary))
    known, scanned = repository_tapes(output)
    rows = []
    stats = {"left_clauses": 0, "complete_reverse_segmentations": 0, "partial_prefix_probes": 0, "exact_closures": 0, "mechanically_admitted": 0}
    for left in LEFTS:
        stats["left_clauses"] += 1
        target = normalize_letters(left)[::-1]
        segmentations, cache_info, best_prefix = decode(target, vocabulary)
        stats["complete_reverse_segmentations"] += len(segmentations)
        if not segmentations:
            right = " ".join(best_prefix["words"])
            text = left.rstrip(".") + "; " + right
            tape = normalize_letters(text)
            checks = mechanical_admission_checks(text, min_letters=39, max_letters=180)
            rows.append({
                "kind": "partial_reverse_prefix_rejection",
                "left_clause": left,
                "right_prefix": right,
                "reverse_target": target,
                "reverse_prefix_letters": best_prefix["consumed"],
                "reverse_prefix_score": best_prefix["score"],
                "rendered": text,
                "letters": len(tape),
                "tape": tape,
                "exact": bool(tape) and tape == tape[::-1],
                "known_tape": tape in known,
                "checks": checks,
                "admitted": False,
                "decoder_cache": cache_info._asdict(),
                "rejection": "no complete reverse lexical segmentation within bounded frontier",
                "reader_status": "not_run; partial probes are not candidates",
            })
            stats["partial_prefix_probes"] += 1
            continue
        for words in segmentations:
            right = " ".join(words)
            text = left.rstrip(".") + "; " + right + "."
            tape = normalize_letters(text)
            checks = mechanical_admission_checks(text, min_letters=39, max_letters=180)
            exact = bool(tape) and tape == tape[::-1]
            if exact:
                stats["exact_closures"] += 1
            admitted = exact and tape not in known and all(checks.values())
            if admitted:
                stats["mechanically_admitted"] += 1
            rows.append({
                "left_clause": left,
                "right_clause": right,
                "rendered": text,
                "reverse_target": target,
                "letters": len(tape),
                "tape": tape,
                "exact": exact,
                "known_tape": tape in known,
                "checks": checks,
                "admitted": admitted,
                "decoder_cache": cache_info._asdict(),
                "reader_status": "not_run; programmatic checks do not certify readability",
            })
    result = {
        "status": "complete_constrained_reverse_lexical_decode",
        "state_space_signature": "reverse-prefix-lexical-v2|memoized-frequency-frontier|four-fresh-lefts|no-brown-pos-semordnilap",
        "config": {"max_words": MAX_WORDS, "max_word_length": MAX_WORD_LENGTH, "frontier_cap": FRONTIER_CAP, "independent_right_lexicalization": True},
        "repository_tapes": len(known),
        "repository_json_files_scanned": scanned,
        "rows": rows,
        "stats": stats,
        "next_operator": "Add a typed semantic clause grammar over only reverse-compatible prefixes; do not widen this lexical frontier or reuse a prior family.",
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source_text_copied": False, "brown_corpus_used": False},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"stats": result["stats"], "rows": len(result["rows"])}))
