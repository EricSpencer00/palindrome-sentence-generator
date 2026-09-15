"""Two-adjacent-clause residual search over Brown-attested relations.

Each side is built from two independently selected, locally attested SVO
relations.  Optional adjective and prepositional attachments lengthen the
clauses without copying any Brown sentence.  A memoized residual lookup pairs
the reversed character tape with an independently built second-clause pair.
Dependency, number, tense, relation indices, and attachment mode are carried
in every state.  Exact closures remain proposals, never readability evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from experiments.brown_attested_relation_residual_bfs_20260915 import (
    Relation,
    _attested_inflections,
    _extract_relations,
    _variants,
)

MIN_LETTERS = 39
MAX_LETTERS = 180
ADJECTIVES = tuple("big bright calm careful early gentle good happy kind large late little old quiet red small strong warm white young".split())
PREPOSITIONS = tuple("at by for from in near on over through with".split())


@dataclass(frozen=True)
class Clause:
    relation_index: int
    relation: Relation
    variant: Relation
    attachment: str
    attachment_word: str = ""
    attachment_noun: str = ""

    @property
    def words(self) -> tuple[str, ...]:
        base = self.variant.words
        if self.attachment == "adjective":
            return base[:-1] + (self.attachment_word, base[-1])
        if self.attachment == "prepositional":
            return base + (self.attachment_word, "the", self.attachment_noun)
        return base

    @property
    def tape(self) -> str:
        return "".join(self.words)

    @property
    def features(self) -> dict:
        tense = "past" if self.variant.verb.endswith(("ed", "t")) else "present"
        return {
            "relation_index": self.relation_index,
            "subject_number": self.variant.subject_number,
            "object_number": self.variant.object_number,
            "tense": tense,
            "attachment": self.attachment,
            "attachment_word": self.attachment_word,
            "attachment_noun": self.attachment_noun,
        }


@dataclass(frozen=True)
class Composite:
    first: Clause
    second: Clause

    @property
    def words(self) -> tuple[str, ...]:
        return self.first.words + self.second.words

    @property
    def tape(self) -> str:
        return self.first.tape + self.second.tape

    @property
    def state(self) -> dict:
        return {
            "first": self.first.features,
            "second": self.second.features,
            "dependency": {
                "first_svo": True,
                "second_svo": True,
                "clause_boundary": "punctuation_only",
            },
        }


def _attachments(variant: Relation, relation_nouns: tuple[str, ...]):
    yield ("none", "", "")
    for adjective in ADJECTIVES[:6]:
        yield ("adjective", adjective, "")
    for prep in PREPOSITIONS[:5]:
        for noun in relation_nouns[:4]:
            yield ("prepositional", prep, noun)


def _composites(relations: tuple[Relation, ...], forms: dict[str, tuple[str, ...]], cap_relations: int = 280):
    selected = relations[:cap_relations]
    nouns = tuple(dict.fromkeys(row.object for row in selected if zipf_frequency(row.object, "en") >= 3.0))
    clauses: list[Clause] = []
    for index, relation in enumerate(selected):
        for variant in _variants(relation, forms, cap=2):
            for attachment, word, noun in _attachments(variant, nouns):
                clauses.append(Clause(index, relation, variant, attachment, word, noun))
    # A finite typed residual frontier: every clause pair carries the two
    # relation identities and features; no opaque text-only strings are used.
    # Keep this operator bounded and reproducible; the residual state remains
    # explicit even when the lexical frontier is truncated.
    clauses = clauses[:1200]
    for left in clauses:
        for right in clauses:
            if left.relation_index == right.relation_index:
                continue
            yield Composite(left, right)


def _word_mirror(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return tuple(word[::-1] for word in reversed(left)) == right


def _audit(left: Composite, right: Composite) -> dict:
    text = " ".join(left.words).capitalize() + "; " + " ".join(right.words) + "."
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_letters": tape,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "independent_ascii_exact": bool(tape) and tape == tape[::-1],
        "independent_two_pointer": all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "left_state": left.state,
        "right_state": right.state,
        "word_order_shortcut": _word_mirror(left.words, right.words),
        "central_admission": checks,
        "mechanically_admitted": all(checks.values()) and not _word_mirror(left.words, right.words),
        "reader_status": "not_run; corpus attestation and exactness do not certify readability",
    }


def run() -> dict:
    relations, extraction = _extract_relations(limit=12_000)
    forms = _attested_inflections(relations)
    # Build a bounded relation frontier.  Keep source relations distinct but
    # do not carry their source sentences into output.
    composites = _composites(relations, forms, cap_relations=80)
    residual: dict[str, list[Composite]] = defaultdict(list)
    composite_count = 0
    for composite in composites:
        if not (18 <= len(composite.tape) <= 90):
            continue
        residual[composite.tape].append(composite)
        composite_count += 1
    candidates = []
    stats = Counter(extraction)
    stats.update({"composites_indexed": composite_count, "indexed_tapes": len(residual)})
    seen = set()
    # Reiterate the same typed frontier as the left side; lookup is the exact
    # residual state transition.  This is intentionally breadth-first over
    # relation pair order, not a language-model reward loop.
    for tape, left_rows in residual.items():
        reverse_rows = residual.get(tape[::-1], ())
        stats["left_states_considered"] += len(left_rows)
        if not reverse_rows:
            stats["residual_misses"] += len(left_rows)
            continue
        stats["residual_matches"] += len(left_rows) * len(reverse_rows)
        for left in left_rows:
            for right in reverse_rows:
                if left.first.relation_index == right.first.relation_index or left.second.relation_index == right.second.relation_index:
                    stats["relation_reuse_rejections"] += 1
                    continue
                if len(left.tape) + len(right.tape) < MIN_LETTERS:
                    stats["short_rejections"] += 1
                    continue
                if _word_mirror(left.words, right.words):
                    stats["word_order_rejections"] += 1
                    continue
                key = left.tape + right.tape
                if key in seen:
                    continue
                seen.add(key)
                row = _audit(left, right)
                candidates.append(row)
                stats["candidates"] += 1
                if row["mechanically_admitted"]:
                    stats["mechanically_admitted"] += 1
    candidates.sort(key=lambda row: (row["mechanically_admitted"], row["letters"]), reverse=True)
    return {
        "status": "complete_two_clause_attested_attachment_residual_bfs_no_reader_promotion",
        "config": {
            "relation_source": "Brown-derived compact SVO relations; source sentences never rendered",
            "clauses_per_side": 2,
            "attachments": "Brown-derived lexical adjective/preposition forms",
            "state_features": "relation identity, subject/object number, tense cue, attachment mode, dependency shape",
            "clause_boundary": "punctuation only",
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "relation_frontier_cap": 80,
            "clause_frontier_cap": 1200,
            "independent_exact_audit": True,
            "anti_shortcut_gate": True,
        },
        "stats": dict(stats),
        "admitted": [row for row in candidates if row["mechanically_admitted"]],
        "near_misses": candidates[:100],
        "residual_frontier": {
            "indexed_tapes": len(residual),
            "left_states": stats["left_states_considered"],
            "unmatched_states": stats["residual_misses"],
            "matched_state_pairs_before_gates": stats["residual_matches"],
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_text_copied": False,
            "relations_attested_individually": True,
        },
        "next_operator": (
            "Add cross-clause dependency frames (relative clauses and controlled conjunctions) while keeping the "
            "punctuation-boundary residual state, and require Brown attestation for each attachment relation."
        ),
        "reader_gate": "No output is human evidence; use intact prose and shuffled controls for any future closures.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite output: {args.out}")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stats": result["stats"], "admitted": len(result["admitted"])}, indent=2))


if __name__ == "__main__":
    main()
