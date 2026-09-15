"""Cross-clause dependency residual search from Brown-attested relations.

The lexical source is unchanged: every content relation is extracted from a
Brown SVO frame.  This branch adds only controlled dependency operators:
relative ``who`` attachments and conjunctions (``and``, ``or``, ``but``).
Both sides are independently composed and matched on reversed character
tapes across a punctuation boundary.  No Brown sentence is copied into an
output, and no closure is reader evidence without a blinded study.
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
CONTROLLED_CONJUNCTIONS = ("and", "or", "but")


@dataclass(frozen=True)
class DependencyFrame:
    head_index: int
    attach_index: int
    head: Relation
    attachment: Relation
    mode: str
    connector: str

    @property
    def words(self) -> tuple[str, ...]:
        head = self.head.words
        tail = self.attachment.words
        if self.mode == "relative":
            return head + ("who",) + tail
        if self.mode == "conjunction":
            return head + (self.connector,) + tail
        return head + ("who",) + tail + (self.connector,)

    @property
    def tape(self) -> str:
        return "".join(self.words)

    @property
    def state(self) -> dict:
        tense_head = "past" if self.head.verb.endswith(("ed", "t")) else "present"
        tense_attach = "past" if self.attachment.verb.endswith(("ed", "t")) else "present"
        return {
            "head_relation_index": self.head_index,
            "attachment_relation_index": self.attach_index,
            "dependency": self.mode,
            "connector": self.connector,
            "head_subject_number": self.head.subject_number,
            "head_object_number": self.head.object_number,
            "attachment_subject_number": self.attachment.subject_number,
            "attachment_object_number": self.attachment.object_number,
            "head_tense": tense_head,
            "attachment_tense": tense_attach,
        }


def _frames(relations: tuple[Relation, ...], forms: dict[str, tuple[str, ...]], cap: int = 72):
    selected = relations[:cap]
    variants: list[tuple[int, Relation]] = []
    for index, relation in enumerate(selected):
        for variant in _variants(relation, forms, cap=1):
            variants.append((index, variant))
    for head_index, head in variants:
        for attach_index, attach in variants:
            if head_index == attach_index:
                continue
            for mode in ("relative", "conjunction"):
                connectors = ("who",) if mode == "relative" else CONTROLLED_CONJUNCTIONS
                for connector in connectors:
                    # The connector is controlled syntax, never a learned or
                    # borrowed lexical item. Content still comes only from
                    # attested SVO relations.
                    yield DependencyFrame(head_index, attach_index, head, attach, mode, connector)


def _word_mirror(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return tuple(word[::-1] for word in reversed(left)) == right


def _audit(left: DependencyFrame, right: DependencyFrame) -> dict:
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
        "reader_status": "not_run; dependency attestation and exactness do not certify readability",
    }


def run() -> dict:
    relations, extraction = _extract_relations(limit=10_000)
    forms = _attested_inflections(relations)
    frames = tuple(_frames(relations, forms, cap=72))
    index: dict[str, list[DependencyFrame]] = defaultdict(list)
    for frame in frames:
        if 18 <= len(frame.tape) <= 90:
            index[frame.tape].append(frame)
    candidates = []
    stats = Counter(extraction)
    stats.update({"dependency_frames": len(frames), "indexed_tapes": len(index)})
    seen = set()
    for tape, lefts in index.items():
        rights = index.get(tape[::-1], ())
        stats["left_states_considered"] += len(lefts)
        if not rights:
            stats["residual_misses"] += len(lefts)
            continue
        stats["residual_matches"] += len(lefts) * len(rights)
        for left in lefts:
            for right in rights:
                if left.head_index == right.head_index or left.attach_index == right.attach_index:
                    stats["relation_reuse_rejections"] += 1
                    continue
                total = len(left.tape) + len(right.tape)
                if total < MIN_LETTERS:
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
        "status": "complete_cross_clause_dependency_residual_bfs_no_reader_promotion",
        "config": {
            "relation_source": "Brown-derived compact SVO relations; source sentences never rendered",
            "dependency_operators": ["relative_who", "controlled_and_or_but"],
            "relation_frontier_cap": 72,
            "state_features": "relation identities, number, tense cue, dependency mode, connector",
            "clause_boundary": "punctuation only",
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "independent_exact_audit": True,
            "anti_shortcut_gate": True,
        },
        "stats": dict(stats),
        "admitted": [row for row in candidates if row["mechanically_admitted"]],
        "near_misses": candidates[:100],
        "residual_frontier": {
            "indexed_tapes": len(index),
            "left_states": stats["left_states_considered"],
            "unmatched_states": stats["residual_misses"],
            "matched_state_pairs_before_gates": stats["residual_matches"],
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_text_copied": False,
            "attachments_individually_attested": True,
        },
        "next_operator": (
            "Add relative-clause attachment with an attested shared subject or object (dependency co-reference), "
            "while keeping controlled connectors and carrying agreement through the residual key."
        ),
        "reader_gate": "No output is human evidence; use intact prose and shuffled controls for any future closure.",
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
