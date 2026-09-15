"""Two-relative chains with attested inner attachments on either edge."""
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
from experiments.brown_shared_coreference_attachment_residual_bfs_20260915 import _attachment_inventory

MIN_LETTERS = 39
MAX_LETTERS = 280


def _tense(verb: str) -> str:
    return "past" if verb.endswith(("ed", "t")) else "present"


@dataclass(frozen=True)
class ChainAttachment:
    head_index: int
    first_index: int
    second_index: int
    head: Relation
    first: Relation
    second: Relation
    edge: str
    attachment: str
    attachment_word: str = ""
    attachment_det: str = ""
    attachment_noun: str = ""
    attachment_prep: str = ""

    @property
    def words(self) -> tuple[str, ...]:
        h, f, s = self.head.words, self.first.words, self.second.words
        if not (len(h) == len(f) == len(s) == 5):
            return ()
        # The two co-reference edges are represented without noun repetition:
        # head subject -> who subject; first object -> that object.
        out = list(h)
        out.extend(("who", f[2], f[3]))
        if self.edge == "first_object":
            if self.attachment == "adjective":
                out.extend((self.attachment_word, f[4]))
            else:
                out.extend((f[4], self.attachment_prep, self.attachment_det, self.attachment_noun))
        else:
            out.append(f[4])
        out.append("that")
        if self.edge == "second_subject" and self.attachment == "adjective":
            out.extend((self.attachment_det, self.attachment_word, s[1], s[2], s[3]))
        else:
            out.extend((s[1], s[2], s[3]))
            if self.edge == "second_subject" and self.attachment == "prepositional":
                out.extend((self.attachment_prep, self.attachment_det, self.attachment_noun))
        return tuple(out)

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
            "attachment_edge": self.edge,
            "attachment_type": self.attachment,
            "attachment_word": self.attachment_word,
            "attachment_prep": self.attachment_prep,
            "attachment_det": self.attachment_det,
            "attachment_noun": self.attachment_noun,
            "head_number": self.head.subject_number,
            "first_object_number": self.first.object_number,
            "head_tense": _tense(self.head.verb),
            "first_tense": _tense(self.first.verb),
            "second_tense": _tense(self.second.verb),
            "agreement_checked": True,
            "attachment_attested": True,
        }


def _chains(relations, forms, adjectives, prepositions, cap=260):
    selected = relations[:cap]
    variants = []
    for index, relation in enumerate(selected):
        for variant in _variants(relation, forms, cap=1):
            if len(variant.words) == 5:
                variants.append((index, variant))
    by_subject: dict[str, list[tuple[int, Relation]]] = defaultdict(list)
    by_object: dict[str, list[tuple[int, Relation]]] = defaultdict(list)
    for index, row in variants:
        by_subject[row.subject].append((index, row))
        by_object[row.object].append((index, row))
    for hi, head in variants:
        for fi, first in by_subject.get(head.subject, ()):
            if fi == hi or first.subject_number != head.subject_number:
                continue
            for si, second in by_object.get(first.object, ()):
                if si in {hi, fi} or second.object_number != first.object_number:
                    continue
                for adjective in adjectives.get(first.object, ())[:4]:
                    yield ChainAttachment(hi, fi, si, head, first, second, "first_object", "adjective", adjective)
                for prep, det in prepositions.get(first.object, ())[:4]:
                    yield ChainAttachment(hi, fi, si, head, first, second, "first_object", "prepositional", "", det, first.object, prep)
                for adjective in adjectives.get(second.subject, ())[:4]:
                    yield ChainAttachment(hi, fi, si, head, first, second, "second_subject", "adjective", adjective, second.subject_det)
                for prep, det in prepositions.get(second.subject, ())[:4]:
                    yield ChainAttachment(hi, fi, si, head, first, second, "second_subject", "prepositional", "", det, second.subject, prep)


def _word_mirror(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return tuple(word[::-1] for word in reversed(left)) == right


def _audit(left: ChainAttachment, right: ChainAttachment) -> dict:
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
    relations, extraction = _extract_relations(limit=14_000)
    forms = _attested_inflections(relations)
    adjectives, prepositions, attachment_sentences = _attachment_inventory()
    frames = tuple(_chains(relations, forms, adjectives, prepositions, cap=len(relations)))
    residual: dict[str, list[ChainAttachment]] = defaultdict(list)
    for frame in frames:
        if 20 <= len(frame.tape) <= 140:
            residual[frame.tape].append(frame)
    stats = Counter(extraction)
    stats.update({"attachment_brown_sentences_scanned": attachment_sentences, "chain_attachment_frames": len(frames), "indexed_tapes": len(residual)})
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
        "status": "complete_two_relative_chain_attachment_residual_bfs_no_reader_promotion",
        "config": {
            "relation_source": "Brown-derived SVO relations; source sentences never rendered",
            "attachment_source": "independently Brown-attested adjective-noun and preposition-noun pairs",
            "dependency_chain": "head subject -> first relative subject; first object -> second relative object",
            "attachment_edges": ["first_relative_object", "second_relative_subject"],
            "state_features": "three relation identities, two co-reference variables, number, tense, attachment metadata",
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
            "Add a bounded relative-chain lexical repair at one content boundary while preserving both "
            "co-reference variables and attachment attestations; independently audit each resulting closure."
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
