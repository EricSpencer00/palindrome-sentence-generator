"""Fresh lexical-bridge search from short complete sentences.

Each source is an intact, task-authored sentence of 30--60 letters.  Its
immutable letter tape is reversed, and the right side is enumerated as words
whose boundaries may cross every source-word boundary.  Those words are then
checked by an independent typed clause parser and the central anti-shortcut
admission gate.  The run never lengthens a source unless an exact, parsed
survivor first exists.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

MIN_SOURCE_LETTERS, MAX_SOURCE_LETTERS = 30, 60
MAX_RIGHT_WORDS, MAX_SEGMENTATIONS = 12, 256
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

SOURCE_SENTENCES = (
    "The careful baker makes fresh bread at home.",
    "A skilled teacher writes clear notes for class.",
    "The patient nurse reads a warm letter aloud.",
    "A quiet artist paints bright walls outdoors.",
    "The kind tailor repairs a fine coat neatly.",
)

VOCABULARY = frozenset(
    word.strip() for word in (ROOT / "data" / "lexicon.txt").read_text().splitlines()
    if re.fullmatch(r"[a-z]+", word.strip()) and (len(word.strip()) >= 2 or word.strip() in {"a", "i"})
)


def reverse_tape(text: str) -> str: return normalize_letters(text)[::-1]


def _trie(vocabulary: frozenset[str]) -> dict:
    root: dict = {}
    for word in vocabulary:
        node = root
        for char in word: node = node.setdefault(char, {})
        node.setdefault("", True)
    return root


def segment_reversed_tape(tape: str, vocabulary: frozenset[str], *, limit: int = MAX_SEGMENTATIONS) -> list[tuple[str, ...]]:
    """Enumerate words from immutable reversed chunks, preserving every letter."""
    trie = _trie(vocabulary); results: list[tuple[str, ...]] = []
    @lru_cache(maxsize=None)
    def visit(position: int, word_count: int) -> tuple[tuple[str, ...], ...]:
        if position == len(tape): return ((),) if 3 <= word_count <= MAX_RIGHT_WORDS else ()
        if word_count >= MAX_RIGHT_WORDS: return ()
        node = trie; found = []
        for end in range(position, len(tape)):
            node = node.get(tape[end])
            if node is None: break
            if "" not in node: continue
            for tail in visit(end + 1, word_count + 1):
                found.append((tape[position:end + 1],) + tail)
                if len(found) >= limit: return tuple(found)
        return tuple(found)
    results.extend(visit(0, 0)); return sorted(set(results))[:limit]


def first_lexical_dead_frontier(tape: str, vocabulary: frozenset[str]) -> dict[str, object] | None:
    """Locate the first reachable tape position with no lexical continuation."""
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for position in range(len(tape)):
        if position not in reachable:
            continue
        options = sorted((word for word in vocabulary if tape.startswith(word, position)), key=lambda word: (len(word), word))
        if not options:
            prefix = tape[:position]
            return {"position": position, "prefix": prefix, "remaining": tape[position:], "reachable_segment": reachable[position], "available_words": []}
        for word in options:
            reachable.setdefault(position + len(word), reachable[position] + (word,))
    if len(tape) not in reachable:
        return {"position": len(tape), "prefix": tape, "remaining": "", "reachable_segment": reachable.get(len(tape), ()), "available_words": []}
    return None


def independent_parse_right(words: tuple[str, ...]) -> bool:
    """Parse a complete right clause with explicit NP--VP--NP semantics."""
    det = {"a", "an", "the", "our"}
    subject_adj = {"careful", "patient", "skilled", "quiet", "kind"}
    subject_noun = {"baker", "teacher", "nurse", "artist", "tailor", "writer", "farmer"}
    verbs = {"makes": {"bread", "notes", "coat"}, "writes": {"notes", "letter"}, "reads": {"letter", "notes"}, "paints": {"walls"}, "repairs": {"coat", "walls"}, "carries": {"water"}, "warms": {"bread"}}
    object_adj = {"fresh", "clear", "warm", "bright", "fine", "plain"}
    object_noun = set().union(*verbs.values())
    if len(words) != 6 or words[0] not in det or words[1] not in subject_adj or words[2] not in subject_noun: return False
    verb = words[3]
    return verb in verbs and words[4] in object_adj and words[5] in verbs[verb]


def audit_candidate(source: str, right_words: tuple[str, ...]) -> dict[str, object]:
    right = " ".join(right_words); rendered = source.rstrip(".") + " " + right.capitalize() + "."
    tape = normalize_letters(rendered); gate = mechanical_admission_checks(rendered, min_letters=MIN_SOURCE_LETTERS, max_letters=260); parsed = independent_parse_right(right_words)
    codes = [key for key, value in gate.items() if not value]
    if not parsed: codes.append("independent_right_clause_reparse_failed")
    return {"source": source, "right": right, "rendered": rendered, "source_tape": normalize_letters(source), "right_tape": normalize_letters(right), "normalized": tape, "letters": len(tape), "independent_exact_audit": {"exact": bool(tape) and tape == tape[::-1], "letters": len(tape), "normalized_sha256": sha256(tape.encode()).hexdigest()}, "independent_right_parse": parsed, "central_admission": gate, "mechanically_admitted": not codes, "rejection_codes": codes, "provenance": "fresh complete source sentence -> immutable reversed tape -> lexical word-boundary segmentation -> independent right-clause parse", "reader_status": "human-unreviewed; programmatic checks do not certify readability"}


def run(*, segmentation_limit: int = MAX_SEGMENTATIONS) -> dict[str, object]:
    vocabulary_hash = sha256("\n".join(sorted(VOCABULARY)).encode()).hexdigest(); source_rows = []; records = []; first_failure = None
    for source_id, source in enumerate(SOURCE_SENTENCES, 1):
        tape = normalize_letters(source); valid_length = MIN_SOURCE_LETTERS <= len(tape) <= MAX_SOURCE_LETTERS; reversed_tape = tape[::-1]; segmentations = segment_reversed_tape(reversed_tape, VOCABULARY, limit=segmentation_limit) if valid_length else []
        row = {"source_id": f"B{source_id:02d}", "source": source, "source_letters": len(tape), "source_complete_sentence": True, "source_length_band": valid_length, "reversed_tape": reversed_tape, "segmentation_count": len(segmentations), "segmentations": [" ".join(words) for words in segmentations], "first_lexical_dead_frontier": first_lexical_dead_frontier(reversed_tape, VOCABULARY) if valid_length else None}; source_rows.append(row)
        if not valid_length:
            if first_failure is None: first_failure = {"source_id": row["source_id"], "reason": "source_outside_30_to_60_letter_preflight"}
            continue
        if not segmentations:
            if first_failure is None: first_failure = {"source_id": row["source_id"], "reason": "no_exact_lexical_word_boundary_segmentation", "reversed_tape": reversed_tape, "dead_frontier": row["first_lexical_dead_frontier"]}
            continue
        for right_words in segmentations: records.append({"source_id": row["source_id"], **audit_candidate(source, right_words)})
    exact = [record for record in records if record["independent_exact_audit"]["exact"]]; parsed = [record for record in exact if record["independent_right_parse"]]; admitted = [record for record in parsed if record["mechanically_admitted"]]
    if first_failure is None and not admitted: first_failure = {"reason": "all lexical bridges fail independent syntax or central admission"}
    return {"status": "fresh_short_sentence_lexical_bridge_search", "operator": "freeze a complete short sentence, reverse its letters, and choose right words from cross-boundary chunks", "config": {"source_min_letters": MIN_SOURCE_LETTERS, "source_max_letters": MAX_SOURCE_LETTERS, "segmentation_limit": segmentation_limit, "max_right_words": MAX_RIGHT_WORDS, "lengthen_only_after_exact_candidate": True, "single_complete_source_sentence": True, "right_boundaries_cross_source_words": True, "independent_right_clause_parse": True, "no_repeated_content_required": True, "central_admission_required": True, "human_readability_required_after_admission": True, "corpus_or_catalogue_source": False}, "source_records": source_rows, "records": records, "exact_candidates": exact, "parsed_exact_candidates": parsed, "mechanically_admitted": admitted, "first_real_bridge_failure": first_failure, "next_repair_operator": "Expand the typed right-clause lexicon or alter one complete source sentence only after recording a new cross-boundary segmentation trace; do not lengthen without an exact parsed survivor.", "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "vocabulary_sha256": vocabulary_hash, "source_material": "task-authored short complete sentences; no catalogue text"}, "reader_facing_test": {"status": "not triggered because no admitted exact survivor", "required_when_triggered": "randomized blinded intact-prose controls and word-shuffled controls", "ratings": ["grammatical", "recoverable intent", "coherent thought"]}}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--segmentation-limit", type=int, default=MAX_SEGMENTATIONS); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(segmentation_limit=args.segmentation_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "sources": len(result["source_records"]), "records": len(result["records"]), "exact": len(result["exact_candidates"]), "admitted": len(result["mechanically_admitted"])}, indent=2))


if __name__ == "__main__": main()
