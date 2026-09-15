"""One-lexeme repair search over attached two-relative chains."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from experiments.brown_attested_relation_residual_bfs_20260915 import _attested_inflections, _extract_relations, _variants
from experiments.brown_shared_coreference_attachment_residual_bfs_20260915 import _attachment_inventory
from experiments.brown_two_relative_chain_attachment_residual_bfs_20260915 import ChainAttachment, _chains

MIN_LETTERS = 39
MAX_LETTERS = 280


def _tense(verb: str) -> str:
    return "past" if verb.endswith(("ed", "t")) else "present"


def _verb_choices(relations, target: str, limit: int = 8) -> tuple[str, ...]:
    desired = _tense(target)
    words = sorted(
        {row.verb for row in relations if _tense(row.verb) == desired and row.verb != target},
        key=lambda word: (-zipf_frequency(word, "en"), word),
    )
    return tuple(words[:limit])


def _repair_frames(frame: ChainAttachment, relations, adjectives, prepositions):
    """Yield one-content-lexeme repairs with explicit provenance."""
    for slot, relation_name in (("head_verb", "head"), ("first_verb", "first"), ("second_verb", "second")):
        relation = getattr(frame, relation_name)
        for new_verb in _verb_choices(relations, relation.verb):
            updated = replace(relation, verb=new_verb)
            repaired = replace(frame, **{relation_name: updated})
            yield repaired, {"slot": slot, "old": relation.verb, "new": new_verb, "source": "Brown-attested verb form", "relation_preserved": False}
    # Attachment repairs retain the same noun and require the replacement pair
    # to be independently attested in Brown.
    if frame.attachment == "adjective":
        noun = frame.first.object if frame.edge == "first_object" else frame.second.subject
        for adjective in adjectives.get(noun, ())[:8]:
            if adjective == frame.attachment_word:
                continue
            yield replace(frame, attachment_word=adjective), {"slot": f"{frame.edge}_adjective", "old": frame.attachment_word, "new": adjective, "source": "Brown-attested adjective-noun pair", "relation_preserved": True}
    if frame.attachment == "prepositional":
        noun = frame.attachment_noun
        for prep, det in prepositions.get(noun, ())[:8]:
            if prep == frame.attachment_prep and det == frame.attachment_det:
                continue
            yield replace(frame, attachment_prep=prep, attachment_det=det), {"slot": f"{frame.edge}_preposition", "old": frame.attachment_prep, "new": prep, "source": "Brown-attested preposition-noun pair", "relation_preserved": True}


def _word_mirror(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return tuple(word[::-1] for word in reversed(left)) == right


def _audit(left: ChainAttachment, right: ChainAttachment, left_repair: dict, right_repair: dict) -> dict:
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
        "left_repair": left_repair,
        "right_repair": right_repair,
        "word_order_shortcut": _word_mirror(left.words, right.words),
        "central_admission": checks,
        "mechanically_admitted": all(checks.values()) and not _word_mirror(left.words, right.words),
        "reader_status": "not_run; lexical repair and exactness do not certify readability",
    }


def run() -> dict:
    relations, extraction = _extract_relations(limit=14_000)
    forms = _attested_inflections(relations)
    adjectives, prepositions, attachment_sentences = _attachment_inventory()
    bases = tuple(_chains(relations, forms, adjectives, prepositions, cap=len(relations)))
    variants: list[tuple[ChainAttachment, dict]] = [(frame, {"slot": "none", "old": "", "new": "", "source": "unrepaired base", "relation_preserved": True}) for frame in bases]
    for frame in bases:
        variants.extend(_repair_frames(frame, relations, adjectives, prepositions))
    residual: dict[str, list[tuple[ChainAttachment, dict]]] = defaultdict(list)
    for frame, repair in variants:
        if 20 <= len(frame.tape) <= 140:
            residual[frame.tape].append((frame, repair))
    stats = Counter(extraction)
    stats.update({"attachment_brown_sentences_scanned": attachment_sentences, "base_frames": len(bases), "repair_variants": len(variants), "indexed_tapes": len(residual)})
    candidates = []
    seen = set()
    for tape, left_rows in residual.items():
        right_rows = residual.get(tape[::-1], ())
        stats["left_states_considered"] += len(left_rows)
        if not right_rows:
            stats["residual_misses"] += len(left_rows)
            continue
        stats["residual_matches"] += len(left_rows) * len(right_rows)
        for left, left_repair in left_rows:
            for right, right_repair in right_rows:
                if left_repair["slot"] == "none" and right_repair["slot"] == "none":
                    stats["unrepaired_rejections"] += 1
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
                row = _audit(left, right, left_repair, right_repair)
                candidates.append(row)
                stats["candidates"] += 1
                if row["mechanically_admitted"]:
                    stats["mechanically_admitted"] += 1
    candidates.sort(key=lambda row: (row["mechanically_admitted"], row["letters"]), reverse=True)
    return {
        "status": "complete_attached_chain_bounded_lexical_repair_no_reader_promotion",
        "config": {
            "base_source": "Brown-attested two-relative chains with independently attested attachments",
            "repair_budget": "exactly one verb or attachment lexeme per side; at least one side repaired",
            "repair_features": "verb tense preserved; attachment noun and attestation preserved",
            "state_features": "co-reference variables, number, tense, relation identities, repair slot/provenance",
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
            "relation_and_attachment_sources": "Brown corpus only",
        },
        "next_operator": (
            "Carry the same one-boundary repair budget into a jointly repaired subject/object pair, but enforce "
            "a Brown-attested dependency frame for the repaired relation before residual matching."
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
