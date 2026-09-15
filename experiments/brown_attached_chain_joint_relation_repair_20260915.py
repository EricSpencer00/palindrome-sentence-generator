"""Joint subject/object repair using only complete Brown-attested relations."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from experiments.brown_attested_relation_residual_bfs_20260915 import _attested_inflections, _extract_relations
from experiments.brown_shared_coreference_attachment_residual_bfs_20260915 import _attachment_inventory
from experiments.brown_two_relative_chain_attachment_residual_bfs_20260915 import ChainAttachment, _chains

MIN_LETTERS = 39
MAX_LETTERS = 280


def _signature(frame: ChainAttachment) -> tuple[str, ...]:
    return (
        frame.head.subject, frame.head.verb, frame.head.object,
        frame.first.subject, frame.first.verb, frame.first.object,
        frame.second.subject, frame.second.verb, frame.second.object,
    )


def _joint_slot_delta(source: ChainAttachment, replacement: ChainAttachment) -> int:
    return sum(a != b for a, b in zip(_signature(source), _signature(replacement)))


def _word_mirror(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return tuple(word[::-1] for word in reversed(left)) == right


def _audit(left: ChainAttachment, right: ChainAttachment, left_source: ChainAttachment, right_source: ChainAttachment) -> dict:
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
        "left_joint_repair_slots": _joint_slot_delta(left_source, left),
        "right_joint_repair_slots": _joint_slot_delta(right_source, right),
        "source_relation_signatures": {"left": _signature(left_source), "right": _signature(right_source)},
        "repaired_relation_signatures": {"left": _signature(left), "right": _signature(right)},
        "word_order_shortcut": _word_mirror(left.words, right.words),
        "central_admission": checks,
        "mechanically_admitted": all(checks.values()) and not _word_mirror(left.words, right.words),
        "reader_status": "not_run; Brown attestation and exactness do not certify readability",
    }


def run() -> dict:
    relations, extraction = _extract_relations(limit=14_000)
    forms = _attested_inflections(relations)
    adjectives, prepositions, attachment_sentences = _attachment_inventory()
    frames = tuple(_chains(relations, forms, adjectives, prepositions, cap=len(relations)))
    # Every replacement is a complete frame built from Brown relations; this
    # avoids the one-lexeme repair's un-attested partial relation.  A pair is a
    # valid joint repair only when at least two subject/object/verb slots differ.
    repaired = []
    for source in frames:
        for candidate in frames:
            delta = _joint_slot_delta(source, candidate)
            if delta >= 2:
                repaired.append((candidate, source, delta))
    residual: dict[str, list[tuple[ChainAttachment, ChainAttachment, int]]] = defaultdict(list)
    for candidate, source, delta in repaired:
        if 20 <= len(candidate.tape) <= 140:
            residual[candidate.tape].append((candidate, source, delta))
    stats = Counter(extraction)
    stats.update({
        "attachment_brown_sentences_scanned": attachment_sentences,
        "attested_base_frames": len(frames),
        "joint_relation_repairs": len(repaired),
        "indexed_tapes": len(residual),
    })
    candidates = []
    seen = set()
    for tape, left_rows in residual.items():
        right_rows = residual.get(tape[::-1], ())
        stats["left_states_considered"] += len(left_rows)
        if not right_rows:
            stats["residual_misses"] += len(left_rows)
            continue
        stats["residual_matches"] += len(left_rows) * len(right_rows)
        for left, left_source, left_delta in left_rows:
            for right, right_source, right_delta in right_rows:
                if left_delta < 2 or right_delta < 2:
                    stats["joint_delta_rejections"] += 1
                    continue
                if _word_mirror(left.words, right.words):
                    stats["word_order_rejections"] += 1
                    continue
                key = left.tape + right.tape
                if key in seen:
                    continue
                seen.add(key)
                row = _audit(left, right, left_source, right_source)
                candidates.append(row)
                stats["candidates"] += 1
                if row["mechanically_admitted"]:
                    stats["mechanically_admitted"] += 1
    candidates.sort(key=lambda row: (row["mechanically_admitted"], row["letters"]), reverse=True)
    return {
        "status": "complete_joint_attested_relation_repair_no_reader_promotion",
        "config": {
            "base_source": "attached two-relative chains built from Brown-attested SVO relations",
            "repair_definition": "replace a complete relation; require >=2 changed content slots across subject/verb/object fields",
            "attachment_attestation": "retained from independently Brown-attested adjective/preposition pair",
            "state_features": "two co-reference variables, number, tense, relation identities, attachment metadata, repair delta",
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
            "complete_repaired_relations_attested": True,
        },
        "next_operator": (
            "Add a jointly repaired relation pair with shared head subject/object variables constrained before "
            "attachment expansion, then retain exact residual matching and independent audit."
        ),
        "reader_gate": "No output is human evidence; future candidates require intact-prose and shuffled-control readers.",
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
