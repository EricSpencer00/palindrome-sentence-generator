"""Phrase-lattice wrapper search with independent reverse segmentation.

This experiment changes the construction unit from words/slots to short
attested phrase spans.  A left span is taken from one Brown sentence window;
the reverse character tape is segmented independently from a held-out lexical
index.  The composed pair is audited as a whole.  Source spans are provenance
only: they are never treated as a generated readability certificate, and any
row containing a catalogue tape remains a control rather than a candidate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import functools
import hashlib
import json
from pathlib import Path

from nltk.corpus import brown
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "attested-phrase-pair-wrapper-20260915"
SIGNATURE = (
    "attested-phrase-pair-wrapper|independent-corpus-ngram-index|"
    "reverse-tape-segmentation|semantic-composition-around-frozen-center|"
    "reader-gated-audit"
)
MIN_LETTERS = 39
MAX_LETTERS = 180
ORDINARY_SHORT = frozenset(
    "ah am an as at be by do go he if in is it me my no of oh on or so to up us we".split()
)


def _lexicon() -> frozenset[str]:
    return frozenset(Path(ROOT / "data/lexicon.txt").read_text().split())


def _word_ok(word: str, lexicon: frozenset[str]) -> bool:
    return (
        word.isascii()
        and word.isalpha()
        and (len(word) >= 3 or word in ORDINARY_SHORT)
        and word in lexicon
        and zipf_frequency(word, "en") >= 3.2
    )


def _phrase_bank(lexicon: frozenset[str]) -> list[dict]:
    """Extract unique, common Brown n-grams as independently sourced spans."""
    spans: dict[str, dict] = {}
    for sentence_id, sentence in enumerate(brown.sents()):
        words = [word.casefold() for word in sentence]
        if not words or any(not _word_ok(word, lexicon) for word in words):
            continue
        for width in range(2, 8):
            for start in range(len(words) - width + 1):
                phrase = tuple(words[start : start + width])
                tape = "".join(phrase)
                if not 10 <= len(tape) <= 42:
                    continue
                spans.setdefault(
                    tape,
                    {
                        "words": phrase,
                        "source": "nltk.brown",
                        "sentence_id": sentence_id,
                        "start": start,
                    },
                )
    return sorted(spans.values(), key=lambda row: (len("".join(row["words"])), row["words"]))


def _segmenter(lexicon: frozenset[str]):
    vocabulary = tuple(sorted((word for word in top_n_list("en", 30_000) if _word_ok(word, lexicon)), key=lambda word: (-zipf_frequency(word, "en"), word)))
    by_first: dict[str, list[str]] = defaultdict(list)
    for word in vocabulary:
        by_first[word[0]].append(word)

    @functools.lru_cache(maxsize=30_000)
    def segment(tape: str) -> tuple[tuple[str, ...], ...]:
        found: list[tuple[str, ...]] = []

        def visit(offset: int, words: tuple[str, ...]) -> None:
            if offset == len(tape):
                found.append(words)
                return
            if len(words) >= 7 or len(found) >= 80:
                return
            for word in by_first.get(tape[offset], ()):
                if word in words or not tape.startswith(word, offset):
                    continue
                visit(offset + len(word), words + (word,))

        visit(0, ())
        return tuple(found)

    return vocabulary, segment


def _audit(text: str) -> dict:
    from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

    tape = normalize_letters(text)
    independent = "".join(ch for ch in text.casefold() if "a" <= ch <= "z")
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_tape": tape,
        "independent_ascii_tape": independent,
        "exact": bool(tape) and tape == tape[::-1],
        "independent_exact": bool(independent) and independent == independent[::-1],
        "tapes_equal": tape == independent,
        "mechanical_checks": checks,
        "mechanically_admitted": bool(tape) and tape == tape[::-1] and all(checks.values()),
        "sha256": hashlib.sha256(tape.encode()).hexdigest(),
    }


def run() -> dict:
    lexicon = _lexicon()
    spans = _phrase_bank(lexicon)
    vocabulary, segment = _segmenter(lexicon)
    rows: list[dict] = []
    probes: list[dict] = []
    stats = Counter(
        phrase_spans=len(spans),
        segment_vocabulary=len(vocabulary),
        reverse_segment_calls=0,
        reverse_segment_hits=0,
        exact=0,
        mechanically_admitted=0,
        reader_eligible=0,
    )
    known = set()
    for raw in json.loads((ROOT / "data/known_palindromes.json").read_text()):
        known.add("".join(ch for ch in raw.casefold() if "a" <= ch <= "z"))
    seen: set[str] = set()
    for span in spans:
        left_words = tuple(span["words"])
        left_tape = "".join(left_words)
        stats["reverse_segment_calls"] += 1
        segmentations = segment(left_tape[::-1])
        if not segmentations:
            if len(probes) < 40:
                probes.append(
                    {
                        "rendered": " ".join(left_words).capitalize() + ".",
                        "letters": len(left_tape),
                        "normalized_tape": left_tape,
                        "required_reverse_tape": left_tape[::-1],
                        "reverse_segment_status": "no independent lexical segmentation",
                        "source_span": span,
                    }
                )
            continue
        stats["reverse_segment_hits"] += len(segmentations)
        for right_words in segmentations:
            if tuple(word[::-1] for word in reversed(left_words)) == right_words:
                continue
            words = left_words + right_words
            if len("".join(words)) < MIN_LETTERS:
                continue
            tape = "".join(words)
            if tape in seen:
                continue
            seen.add(tape)
            text = " ".join(left_words).capitalize() + "; " + " ".join(right_words) + "."
            audit = _audit(text)
            if tape in known:
                audit["catalogue_control"] = True
                audit["mechanically_admitted"] = False
            else:
                audit["catalogue_control"] = False
            stats["exact"] += int(audit["exact"])
            stats["mechanically_admitted"] += int(audit["mechanically_admitted"])
            rows.append(
                {
                    "rendered": text,
                    "left_span": span,
                    "right_words": list(right_words),
                    "audit": audit,
                    "reader_status": "not_run; attested spans and programmatic checks cannot certify readability",
                }
            )
    rows.sort(key=lambda row: (row["audit"]["mechanically_admitted"], row["audit"]["letters"], sum(zipf_frequency(word, "en") for word in row["right_words"])), reverse=True)
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_no_reader_promotion" if not any(row["audit"]["mechanically_admitted"] for row in rows) else "exact_hits_pending_blinded_readers",
        "config": {
            "phrase_widths": [2, 3, 4, 5, 6, 7],
            "left_tape_letters": [10, 42],
            "segment_max_words": 7,
            "catalogue_spans_used_as_generated": False,
            "frozen_center": False,
        },
        "stats": {**stats, "rendered_rows": len(rows)},
        "rendered_candidates": rows[:120],
        "rendered_probes": probes,
        "exact_candidates": [row for row in rows if row["audit"]["mechanically_admitted"]],
        "next_repair": "Replace attested phrase spans with independently authored complete clause frames while retaining reverse segmentation and source provenance; do not import catalogue text or replay a word-level beam.",
        "provenance": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "lexicon": "repository lexicon.txt plus wordfreq top-30k filter",
            "left_source": "NLTK Brown contiguous spans",
            "right_source": "independent word trie; no intact right phrase copied",
            "programmatic_readability_claim": False,
        },
        "reader_gate": "No row is reader evidence; only a manually inspected intact-prose survivor may enter randomized blinded intact/shuffled reading.",
    }


if __name__ == "__main__":
    out = run()
    path = ROOT / "runs" / "attested-phrase-pair-wrapper-20260915.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "stats": out["stats"], "path": str(path)}, indent=2))
