"""Agreement-aware shared-coreference relative-clause residual search.

Frames are assembled only from Brown-attested SVO relations.  A relative
clause shares either the head subject or head object, with number agreement
and tense carried in the state.  Independently authored dependency frames are
matched on reversed character tapes across punctuation; source sentences are
never copied into outputs.
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
MAX_LETTERS = 180


def _tense(verb: str) -> str:
    return "past" if verb.endswith(("ed", "t")) else "present"


@dataclass(frozen=True)
class CorefFrame:
    head_index: int
    relative_index: int
    head: Relation
    relative: Relation
    coreference: str

    @property
    def words(self) -> tuple[str, ...]:
        h = self.head.words
        r = self.relative.words
        if self.coreference == "shared_subject":
            # The relative subject is the head subject, so omit it after who.
            return h[:1] + (h[1], "who", r[2], r[3], r[4]) if len(h) == 5 and len(r) == 5 else h + ("who",) + r
        # Object relative: ``the man sees a dog that the woman likes``.
        if len(h) == 5 and len(r) == 5:
            return h + ("that", r[1], r[2], r[3], r[4])
        return h + ("that",) + r

    @property
    def tape(self) -> str:
        return "".join(self.words)

    @property
    def state(self) -> dict:
        return {
            "head_relation_index": self.head_index,
            "relative_relation_index": self.relative_index,
            "coreference": self.coreference,
            "head_subject_number": self.head.subject_number,
            "head_object_number": self.head.object_number,
            "relative_subject_number": self.relative.subject_number,
            "relative_object_number": self.relative.object_number,
            "head_tense": _tense(self.head.verb),
            "relative_tense": _tense(self.relative.verb),
            "agreement_checked": True,
        }


def _frames(relations: tuple[Relation, ...], forms: dict[str, tuple[str, ...]], cap: int = 180):
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
        # Relative subject shares the head subject.  Agreement is carried as a
        # hard condition, not a post-hoc score.
        for relative_index, relative in by_subject.get(head.subject, ()):
            if relative_index != head_index and relative.subject_number == head.subject_number:
                yield CorefFrame(head_index, relative_index, head, relative, "shared_subject")
        # Relative object shares the head object.
        for relative_index, relative in by_object.get(head.object, ()):
            if relative_index != head_index and relative.object_number == head.object_number:
                yield CorefFrame(head_index, relative_index, head, relative, "shared_object")


def _word_mirror(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return tuple(word[::-1] for word in reversed(left)) == right


def _audit(left: CorefFrame, right: CorefFrame) -> dict:
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
        "reader_status": "not_run; agreement and exactness do not certify readability",
    }


def run() -> dict:
    relations, extraction = _extract_relations(limit=12_000)
    forms = _attested_inflections(relations)
    frames = tuple(_frames(relations, forms, cap=180))
    residual: dict[str, list[CorefFrame]] = defaultdict(list)
    for frame in frames:
        if 18 <= len(frame.tape) <= 90:
            residual[frame.tape].append(frame)
    candidates = []
    stats = Counter(extraction)
    stats.update({"coreference_frames": len(frames), "indexed_tapes": len(residual)})
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
        "status": "complete_shared_coreference_relative_residual_bfs_no_reader_promotion",
        "config": {
            "relation_source": "Brown-derived compact SVO relations; source sentences never rendered",
            "coreference_modes": ["shared_subject", "shared_object"],
            "agreement": "subject/object number equality required at frame construction",
            "state_features": "relation identities, co-reference mode, subject/object number, tense",
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
            "Retain shared-coreference state while permitting an independently attested adjective or prepositional "
            "attachment inside the relative clause; preserve agreement and tense as hard residual constraints."
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
