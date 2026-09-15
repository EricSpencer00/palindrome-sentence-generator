"""Two-relative-clause chain with shared agreement variables.

The chain is built from independently Brown-attested SVO relations.  The
first relative clause shares the head subject; the second shares the first
relative's object.  Shared nouns are represented by relative syntax (``who``
and ``that``), not copied into the surface string.  Number and tense remain
explicit hard state features.  Reversed character tapes are matched between
independently composed chains across punctuation.
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
MAX_LETTERS = 240


def _tense(verb: str) -> str:
    return "past" if verb.endswith(("ed", "t")) else "present"


@dataclass(frozen=True)
class RelativeChain:
    head_index: int
    first_index: int
    second_index: int
    head: Relation
    first: Relation
    second: Relation

    @property
    def words(self) -> tuple[str, ...]:
        h = self.head.words
        f = self.first.words
        s = self.second.words
        if not (len(h) == len(f) == len(s) == 5):
            return ()
        # h subject is the implicit subject of ``who ...``; f object is the
        # implicit object of ``that ...``.  Neither shared content noun is
        # duplicated in the output.
        return h + ("who", f[2], f[3], f[4], "that", s[1], s[2], s[3])

    @property
    def tape(self) -> str:
        return "".join(self.words)

    @property
    def state(self) -> dict:
        return {
            "head_relation_index": self.head_index,
            "first_relative_index": self.first_index,
            "second_relative_index": self.second_index,
            "dependency_edges": [
                {"edge": "head_subject_to_first_subject", "variable": self.head.subject, "number": self.head.subject_number},
                {"edge": "first_object_to_second_object", "variable": self.first.object, "number": self.first.object_number},
            ],
            "head_subject_number": self.head.subject_number,
            "first_subject_number": self.first.subject_number,
            "first_object_number": self.first.object_number,
            "second_object_number": self.second.object_number,
            "head_tense": _tense(self.head.verb),
            "first_tense": _tense(self.first.verb),
            "second_tense": _tense(self.second.verb),
            "agreement_checked": True,
        }


def _chains(relations: tuple[Relation, ...], forms: dict[str, tuple[str, ...]], cap: int = 260):
    selected = relations[:cap]
    variants = []
    for index, relation in enumerate(selected):
        for variant in _variants(relation, forms, cap=1):
            if len(variant.words) == 5:
                variants.append((index, variant))
    by_subject: dict[str, list[tuple[int, Relation]]] = defaultdict(list)
    by_object: dict[str, list[tuple[int, Relation]]] = defaultdict(list)
    for index, variant in variants:
        by_subject[variant.subject].append((index, variant))
        by_object[variant.object].append((index, variant))
    for head_index, head in variants:
        for first_index, first in by_subject.get(head.subject, ()):
            if first_index == head_index or first.subject_number != head.subject_number:
                continue
            # The second relative is an object-relative clause: its object is
            # the first relative's object, with number agreement enforced.
            for second_index, second in by_object.get(first.object, ()):
                if second_index in {head_index, first_index} or second.object_number != first.object_number:
                    continue
                chain = RelativeChain(head_index, first_index, second_index, head, first, second)
                if chain.words:
                    yield chain


def _word_mirror(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return tuple(word[::-1] for word in reversed(left)) == right


def _audit(left: RelativeChain, right: RelativeChain) -> dict:
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
    relations, extraction = _extract_relations(limit=14_000)
    forms = _attested_inflections(relations)
    chains = tuple(_chains(relations, forms, cap=len(relations)))
    residual: dict[str, list[RelativeChain]] = defaultdict(list)
    for chain in chains:
        if 18 <= len(chain.tape) <= 120:
            residual[chain.tape].append(chain)
    stats = Counter(extraction)
    stats.update({"two_relative_chains": len(chains), "indexed_tapes": len(residual)})
    candidates = []
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
        "status": "complete_two_relative_chain_residual_bfs_no_reader_promotion",
        "config": {
            "relation_source": "Brown-derived compact SVO relations; source sentences never rendered",
            "dependency_chain": "head subject -> first relative subject; first object -> second relative object",
            "agreement": "subject/object number equality required on both edges",
            "state_features": "three relation identities, two co-reference variables, number, tense",
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
            "Permit independently attested adjective or prepositional material on either relative edge while "
            "retaining both co-reference variables, agreement, tense, and independent residual matching."
        ),
        "reader_gate": "No output is human evidence; future closures require intact-prose and shuffled-control readers.",
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
