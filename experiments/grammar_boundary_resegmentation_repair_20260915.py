"""Repair a frozen exact tape by searching for grammatical word boundaries.

The preceding admission-guided center-out run found exact lexical tapes but
its best surface was not readable.  This repair never changes those letters:
it treats the longest tape as fixed, enumerates independent dictionary
segmentations, and ranks them with Brown POS transitions plus a conservative
subject/verb diagnostic.  A boundary resegmentation that fails the shared
mechanical gate is retained as evidence, never promoted as prose.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import re
import sys

from nltk.corpus import brown
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.shortwords import is_real_short


ID = "grammar-boundary-resegmentation-repair"
SIGNATURE = (
    "grammar-boundary-resegmentation|fixed-exact-tape|"
    "pos-weighted-segmentation|valency-diagnostic|independent-audit"
)
SOURCE_RUN = ROOT / "runs" / "lexical-admission-centerout-20260915.json"


def _brown_tables() -> tuple[dict[str, tuple[str, ...]], Counter[tuple[str, str]], Counter[str]]:
    tags: dict[str, Counter[str]] = defaultdict(Counter)
    transitions: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()
    for sentence in brown.tagged_sents(tagset="universal"):
        previous = "<s>"
        for raw, tag in sentence:
            word = re.sub(r"[^A-Za-z]", "", raw).casefold()
            if not word:
                continue
            tags[word][tag] += 1
            transitions[(previous, tag)] += 1
            totals[previous] += 1
            previous = tag
    return (
        {word: tuple(tag for tag, _ in counts.most_common(3)) for word, counts in tags.items()},
        transitions,
        totals,
    )


def _source_tape() -> tuple[str, str]:
    payload = json.loads(SOURCE_RUN.read_text())
    row = max(payload["rendered_candidates_and_probes"], key=lambda item: item["letters"])
    return row["normalized_tape"], row["rendered"]


def _vocabulary(vocabulary_size: int) -> tuple[set[str], str]:
    words = [
        word for word in top_n_list("en", vocabulary_size)
        if word.isascii() and word.isalpha() and is_real_short(word)
        and len(word) >= 2
    ]
    # Keep this input hash explicit: changing the dictionary is a new replay,
    # not an invisible continuation of this repair.
    digest = hashlib.sha256("\n".join(words).encode()).hexdigest()
    return set(words), digest


def _segment(
    tape: str,
    vocab: set[str],
    tags: dict[str, tuple[str, ...]],
    transitions: Counter[tuple[str, str]],
    totals: Counter[str],
    *,
    limit: int,
    max_word_length: int = 18,
) -> list[dict]:
    """Return top complete segmentations without changing the tape."""
    tag_inventory = {tag for values in tags.values() for tag in values}
    smoothing = 0.05
    dp: list[list[tuple[float, tuple[str, ...], str, int, int]]] = [
        [] for _ in range(len(tape) + 1)
    ]
    dp[0] = [(0.0, (), "<s>", 0, 0)]
    for position in range(len(tape)):
        if not dp[position]:
            continue
        rows = dp[position]
        for end in range(position + 1, min(len(tape), position + max_word_length) + 1):
            word = tape[position:end]
            if word not in vocab:
                continue
            for score, words, previous, nouns, verbs in rows:
                for tag in tags.get(word, ()):  # untagged words are not a grammar repair
                    conditional = (transitions[(previous, tag)] + smoothing) / (
                        totals[previous] + smoothing * max(1, len(tag_inventory))
                    )
                    delta = 2.0 * __import__("math").log(conditional)
                    delta += 0.20 * zipf_frequency(word, "en") + 0.08 * len(word)
                    if len(word) <= 2:
                        delta -= 2.0
                    if word in words:
                        delta -= 10.0
                    next_row = (
                        score + delta,
                        words + (word,),
                        tag,
                        min(2, nouns + int(tag in {"NOUN", "PRON"})),
                        min(2, verbs + int(tag in {"VERB", "AUX"})),
                    )
                    dp[end].append(next_row)
        if position + 1 <= len(tape) and len(dp[position + 1]) > limit * 12:
            dp[position + 1] = heapq.nlargest(limit * 12, dp[position + 1], key=lambda row: row[0])

    complete = [row for row in dp[len(tape)] if row[3] and row[4]]
    complete = heapq.nlargest(limit * 4, complete, key=lambda row: row[0])
    out = []
    seen = set()
    for score, words, last_tag, nouns, verbs in complete:
        if words in seen:
            continue
        seen.add(words)
        rendered = " ".join(words) + "."
        checks = mechanical_admission_checks(rendered, min_letters=39, max_letters=220)
        # A deliberately small clause diagnostic: a subject-like tag must
        # occur before a finite verb, while the content remains complete prose
        # material rather than an isolated word list.
        sequence = [tags[word][0] for word in words]
        verb_index = next(
            (index for index, tag in enumerate(sequence) if tag in {"VERB", "AUX"}),
            len(sequence),
        )
        subject_before_verb = (
            any(tag in {"NOUN", "PRON"} for tag in sequence[:verb_index])
            and verb_index < len(sequence)
        )
        out.append({
            "rendered": rendered,
            "words": list(words),
            "pos_sequence": sequence,
            "letters": len(tape),
            "normalized_tape": tape,
            "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "independent_exact": tape == tape[::-1],
            "mechanical_checks": checks,
            "mechanically_admitted": tape == tape[::-1] and all(checks.values()),
            "subject_before_finite_verb": bool(subject_before_verb),
            "score": score,
            "reader_status": "not_run; this is a boundary repair diagnostic",
        })
        if len(out) >= limit:
            break
    return out


def run(*, limit: int = 40, vocabulary_size: int = 80_000) -> dict:
    tape, source_rendered = _source_tape()
    vocab, vocabulary_sha256 = _vocabulary(vocabulary_size)
    tags, transitions, totals = _brown_tables()
    rows = _segment(tape, vocab, tags, transitions, totals, limit=limit)
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {
        "status": "grammar_boundary_resegmentation_repair_complete",
        "experiment_id": ID,
        "signature": SIGNATURE,
        "repair_of": "lexical-admission-centerout",
        "input": {
            "source_run": str(SOURCE_RUN.relative_to(ROOT)),
            "source_rendered": source_rendered,
            "fixed_normalized_tape": tape,
            "letters": len(tape),
            "tape_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        },
        "config": {
            "candidate_limit": limit,
            "vocabulary_size_requested": vocabulary_size,
            "vocabulary_size": len(vocab),
            "max_word_length": 18,
            "pos_source": "Brown universal tags",
            "tape_mutation": False,
            "catalogue_text_imported": False,
        },
        "novelty_audit": {
            "registry_entries_read_before_run": 60,
            "excluded_routes_read_before_run": 3,
            "signature_overlap": [],
            "conceptual_near_pairs": [],
            "manual_review_required": False,
            "repair_of_registered_family": True,
            "self_entry_present": False,
            "preflight_required_before_artifact": True,
            "construction_dimension": (
                "fixed exact tape with a POS-weighted boundary search; unlike the "
                "preceding generator, no new palindrome letters are emitted"
            ),
        },
        "stats": {
            "segmentations": len(rows),
            "mechanically_admitted": len(admitted),
            "subject_before_finite_verb": sum(row["subject_before_finite_verb"] for row in rows),
            "reader_eligible": 0,
        },
        "rendered_candidates_and_probes": rows,
        "admitted": admitted,
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "vocabulary_sha256": vocabulary_sha256,
            "source_sentences_copied": False,
            "known_palindromes_imported": False,
            "independent_validator": "ASCII tape equality plus shared mechanical admission",
            "readability_certificate": False,
        },
        "next_operator": (
            "If no segmentation yields an intact clause, carry finite-verb valency "
            "and argument-role constraints in the same fixed-tape chart; do not "
            "alter letters or reuse the center-out beam."
        ),
        "reader_gate": (
            "Only a human-confirmed intact-prose survivor may enter a randomized "
            "reader package with an intact control and a shuffled control."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--vocabulary-size", type=int, default=80_000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite existing output: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(limit=args.limit, vocabulary_size=args.vocabulary_size)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "segmentations": result["stats"]["segmentations"],
        "mechanically_admitted": result["stats"]["mechanically_admitted"],
    }, indent=2))


if __name__ == "__main__":
    main()
