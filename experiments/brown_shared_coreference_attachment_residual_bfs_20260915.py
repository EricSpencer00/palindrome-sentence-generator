"""Shared-coreference relative search with attested inner attachments.

This is the next constructive branch after the agreement-aware relative
search.  Relation content comes from Brown SVO frames; adjective--noun and
preposition--noun attachments are separately attested in Brown and are never
copied as sentence text.  Frames are matched independently on reversed tapes
across punctuation, then exact and anti-shortcut audited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

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
MAX_LETTERS = 220
DETS = frozenset({"a", "an", "the"})


def _attachment_inventory() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[tuple[str, str], ...]], int]:
    """Extract independently attested ADJ-NOUN and ADP-(DET)-NOUN pairs."""
    from nltk.corpus import brown

    adjectives: dict[str, set[str]] = defaultdict(set)
    prepositions: dict[str, set[tuple[str, str]]] = defaultdict(set)
    sentences = 0
    for tagged in brown.tagged_sents(tagset="universal"):
        sentences += 1
        words = [word.casefold() for word, _ in tagged]
        tags = [tag for _, tag in tagged]
        for i in range(len(words) - 1):
            if tags[i] == "ADJ" and tags[i + 1] == "NOUN" and words[i].isalpha() and words[i + 1].isalpha():
                adjectives[words[i + 1]].add(words[i])
        for i in range(len(words) - 2):
            if tags[i] != "ADP":
                continue
            if tags[i + 1] == "DET" and words[i + 1] in DETS and tags[i + 2] == "NOUN":
                prepositions[words[i + 2]].add((words[i], words[i + 1]))
            elif tags[i + 1] == "NOUN":
                prepositions[words[i + 1]].add((words[i], ""))
    return (
        {noun: tuple(sorted(values)) for noun, values in adjectives.items()},
        {noun: tuple(sorted(values)) for noun, values in prepositions.items()},
        sentences,
    )


def _tense(verb: str) -> str:
    return "past" if verb.endswith(("ed", "t")) else "present"


@dataclass(frozen=True)
class AttachedCoref:
    head_index: int
    relative_index: int
    head: Relation
    relative: Relation
    coreference: str
    attachment: str
    attachment_word: str
    attachment_det: str = ""
    attachment_noun: str = ""
    attachment_prep: str = ""

    @property
    def words(self) -> tuple[str, ...]:
        h = self.head.words
        r = self.relative.words
        if self.coreference == "shared_subject":
            # ``the artist who reads the red book`` or a PP-attached relative.
            words = h[:2] + ("who", r[2], r[3])
            if self.attachment == "adjective":
                return words + (self.attachment_word, r[4])
            return words + (r[4], self.attachment_prep, self.attachment_det, self.attachment_noun)
        # Object coreference repeats the shared object in the relative clause;
        # this makes the dependency explicit and agreement auditable.
        prefix = h + ("that", r[1], r[2], r[3])
        if self.attachment == "adjective":
            return prefix + (self.attachment_word, h[4])
        return prefix + (h[4],)

    @property
    def tape(self) -> str:
        return "".join(self.words)

    @property
    def state(self) -> dict:
        return {
            "head_relation_index": self.head_index,
            "relative_relation_index": self.relative_index,
            "coreference": self.coreference,
            "attachment": self.attachment,
            "attachment_word": self.attachment_word,
            "attachment_prep": self.attachment_prep,
            "attachment_det": self.attachment_det,
            "attachment_noun": self.attachment_noun,
            "head_subject_number": self.head.subject_number,
            "head_object_number": self.head.object_number,
            "relative_subject_number": self.relative.subject_number,
            "relative_object_number": self.relative.object_number,
            "head_tense": _tense(self.head.verb),
            "relative_tense": _tense(self.relative.verb),
            "agreement_checked": True,
            "attachment_attested": True,
        }


def _frames(
    relations: tuple[Relation, ...],
    forms: dict[str, tuple[str, ...]],
    adjectives: dict[str, tuple[str, ...]],
    prepositions: dict[str, tuple[tuple[str, str], ...]],
    cap: int = 180,
):
    selected = relations[:cap]
    variants = []
    for index, relation in enumerate(selected):
        for variant in _variants(relation, forms, cap=1):
            variants.append((index, variant))
    by_subject: dict[str, list[tuple[int, Relation]]] = defaultdict(list)
    by_object: dict[str, list[tuple[int, Relation]]] = defaultdict(list)
    for index, relation in variants:
        by_subject[relation.subject].append((index, relation))
        by_object[relation.object].append((index, relation))
    for head_index, head in variants:
        # The attachment renderer below uses explicit determiner-bearing NPs;
        # reject bare relations rather than silently weakening co-reference.
        if len(head.words) != 5:
            continue
        for relative_index, relative in by_subject.get(head.subject, ()):
            if relative_index == head_index or relative.subject_number != head.subject_number:
                continue
            if len(relative.words) != 5:
                continue
            for adjective in adjectives.get(relative.object, ())[:5]:
                yield AttachedCoref(head_index, relative_index, head, relative, "shared_subject", "adjective", adjective)
            for prep, det in prepositions.get(relative.object, ())[:5]:
                yield AttachedCoref(head_index, relative_index, head, relative, "shared_subject", "prepositional", "", det, relative.object, prep)
        for relative_index, relative in by_object.get(head.object, ()):
            if relative_index == head_index or relative.object_number != head.object_number:
                continue
            if len(relative.words) != 5:
                continue
            # A modifying adjective remains inside the object relative clause.
            for adjective in adjectives.get(head.object, ())[:5]:
                yield AttachedCoref(head_index, relative_index, head, relative, "shared_object", "adjective", adjective)


def _word_mirror(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return tuple(word[::-1] for word in reversed(left)) == right


def _audit(left: AttachedCoref, right: AttachedCoref) -> dict:
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
        "reader_status": "not_run; attachment attestation and exactness do not certify readability",
    }


def run() -> dict:
    relations, extraction = _extract_relations(limit=12_000)
    forms = _attested_inflections(relations)
    adjectives, prepositions, attachment_sentences = _attachment_inventory()
    frames = tuple(_frames(relations, forms, adjectives, prepositions, cap=180))
    residual: dict[str, list[AttachedCoref]] = defaultdict(list)
    for frame in frames:
        if 18 <= len(frame.tape) <= 110:
            residual[frame.tape].append(frame)
    candidates = []
    stats = Counter(extraction)
    stats.update({"attachment_brown_sentences_scanned": attachment_sentences, "coref_attachment_frames": len(frames), "indexed_tapes": len(residual)})
    seen = set()
    for tape, lefts in residual.items():
        rights = residual.get(tape[::-1], ())
        stats["left_states_considered"] += len(lefts)
        if not rights:
            stats["residual_misses"] += len(lefts)
            continue
        stats["residual_matches"] += len(lefts) * len(rights)
        for left in lefts:
            for right in rights:
                if left.head_index == right.head_index or left.relative_index == right.relative_index:
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
        "status": "complete_shared_coreference_attachment_residual_bfs_no_reader_promotion",
        "config": {
            "relation_source": "Brown-derived SVO relations; source sentences never rendered",
            "attachment_source": "independently Brown-attested adjective-noun and preposition-noun pairs",
            "coreference_modes": ["shared_subject", "shared_object"],
            "state_features": "relation identities, co-reference, number, tense, attachment type and lexeme",
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
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
            "relations_and_attachments_attested_individually": True,
        },
        "next_operator": (
            "Retain attested inner attachments but add a bounded two-relative-clause chain, carrying shared "
            "agreement variables through both dependency edges and retaining independent residual matching."
        ),
        "reader_gate": "No output is human evidence; any future closure requires intact-prose and shuffled-control readers.",
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
