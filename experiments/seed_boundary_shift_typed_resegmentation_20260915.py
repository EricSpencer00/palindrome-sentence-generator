"""Typed boundary-shift resegmentation around the 38-letter seed.

The seed contributes only a starting character tape.  For each one-letter
mirrored insertion, both halves are resegmented independently into typed
clause templates.  At least two seed content words must disappear, and the
result cannot retain the seed as a proper span.  This is intentionally a
bounded constructive probe: a zero is useful only if it records exact
closures, near-misses, and the next operator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

SEED_TEXT = "An aide rips nine memos; some men inspire Diana."
SEED_TAPE = normalize_letters(SEED_TEXT)
SEED_CONTENT = frozenset({"aide", "rips", "nine", "memos", "some", "men", "inspire", "diana"})
SHORT = frozenset("a an the i he she we you it they is are was were am be do did can to of in on at for with by from and but or not".split())


def _brown_roles() -> dict[str, tuple[str, ...]]:
    """Build a small fixed common-role lexicon from Brown majority tags."""
    try:
        from nltk.corpus import brown

        counts: dict[str, Counter[str]] = {}
        for word, tag in brown.tagged_words(tagset="universal"):
            word = word.casefold()
            if word.isalpha() and 2 <= len(word) <= 10:
                counts.setdefault(word, Counter())[tag] += 1
        roles: dict[str, list[str]] = {"DET": [], "PRON": [], "ADJ": [], "NOUN": [], "VERB": [], "ADP": [], "ADV": []}
        for word, tags in counts.items():
            role = max(tags, key=tags.get)
            if role in roles and zipf_frequency(word, "en") >= 3.0:
                roles[role].append(word)
        # Keep the inventory bounded and deterministic.  Exclude seed content
        # words so a result cannot merely preserve the seed's prose.
        for role in roles:
            roles[role] = sorted(set(roles[role]) - SEED_CONTENT, key=lambda w: (-zipf_frequency(w, "en"), w))[:320]
        roles["DET"] = [word for word in roles["DET"] if word in {"a", "an", "the", "this", "that"}] or ["a", "an", "the"]
        roles["PRON"] = [word for word in roles["PRON"] if word in {"i", "he", "she", "we", "you", "it", "they"}] or ["i", "he", "she", "we", "you", "it"]
        return {key: tuple(value) for key, value in roles.items()}
    except Exception:
        return {
            "DET": ("a", "an", "the"),
            "PRON": ("i", "he", "she", "we", "you", "it"),
            "ADJ": ("calm", "bright", "young", "quiet", "small", "kind"),
            "NOUN": ("artist", "child", "friend", "guard", "horse", "river", "story", "teacher"),
            "VERB": ("asks", "calls", "draws", "helps", "keeps", "likes", "needs", "sees"),
            "ADP": ("in", "on", "at", "by", "for", "with"),
            "ADV": ("often", "now", "well", "still", "just"),
        }


TEMPLATES = (
    ("det_adj_noun_verb_det_noun", ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN")),
    ("pron_verb_det_noun_adv", ("PRON", "VERB", "DET", "NOUN", "ADV")),
    ("det_noun_verb_adp_det_noun", ("DET", "NOUN", "VERB", "ADP", "DET", "NOUN")),
    ("pron_verb_adv_adp_noun", ("PRON", "VERB", "ADV", "ADP", "NOUN")),
)


def mirrored_insert(tape: str, position: int, letter: str) -> str:
    left = tape[:position] + letter + tape[position:]
    pivot = len(left)
    return left + left[::-1]


def _segments_for_tape(tape: str, roles: dict[str, tuple[str, ...]], slots: tuple[str, ...], cap: int = 160):
    @lru_cache(maxsize=None)
    def rec(offset: int, slot: int):
        if slot == len(slots):
            return ((),) if offset == len(tape) else ()
        remaining_slots = len(slots) - slot - 1
        output = []
        for word in roles[slots[slot]]:
            if not tape.startswith(word, offset):
                continue
            for tail in rec(offset + len(word), slot + 1):
                output.append((word,) + tail)
                if len(output) >= cap:
                    return tuple(output)
        return tuple(output)
    return rec(0, 0)


def _bigram_score(words: tuple[str, ...]) -> float:
    # Diagnostic ranking only; no readability certification.
    return sum(zipf_frequency(a + " " + b, "en") for a, b in zip(words, words[1:]))


def _audit(text: str, operation: dict, left: tuple[str, ...], right: tuple[str, ...]) -> dict:
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=100)
    seed_overlap = sorted(set(left + right) & SEED_CONTENT)
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_letters": tape,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "left_words": list(left),
        "right_words": list(right),
        "operation": operation,
        "seed_content_overlap": seed_overlap,
        "changed_seed_content_words": len(SEED_CONTENT - set(left + right)),
        "independent_exact": bool(tape) and tape == tape[::-1],
        "independent_two_pointer": all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "central_admission": checks,
        "mechanically_admitted": all(checks.values()) and len(SEED_CONTENT - set(left + right)) >= 2,
        "reader_status": "not_run; programmatic filters do not certify readability",
    }


def run() -> dict:
    roles = _brown_roles()
    rows: list[dict] = []
    stats = Counter()
    seen = set()
    failures: list[dict] = []
    # The insertion is mirrored into the full tape; left and right are then
    # independently parsed, so no seed word order or seed interior is reused.
    for position in range(len(SEED_TAPE) // 2 + 1):
        for letter in "abcdefghijklmnopqrstuvwxyz":
            tape = mirrored_insert(SEED_TAPE[: len(SEED_TAPE) // 2], position, letter)
            stats["target_tapes"] += 1
            for left_name, left_slots in TEMPLATES:
                lefts = _segments_for_tape(tape[: len(tape) // 2], roles, left_slots)
                if not lefts:
                    if len(failures) < 40:
                        failures.append({
                            "reason": "no_typed_left_clause_segmentation",
                            "insertion_position": position,
                            "inserted_letter": letter,
                            "left_template": left_name,
                            "left_tape": tape[: len(tape) // 2],
                            "left_letters": len(tape) // 2,
                            "independent_target_exact": tape == tape[::-1],
                        })
                    stats["typed_left_failures"] += 1
                    continue
                stats["left_segmentable_tapes"] += 1
                for right_name, right_slots in TEMPLATES:
                    rights = _segments_for_tape(tape[len(tape) // 2 :], roles, right_slots)
                    if not rights:
                        continue
                    stats["joint_segmentations"] += len(lefts) * len(rights)
                    for left in lefts:
                        for right in rights:
                            words = left + right
                            if set(words) & SEED_CONTENT:
                                # Any retained seed content is evidence, not a
                                # promotion; this probe seeks joint replacement.
                                continue
                            text = " ".join(left) + "; " + " ".join(right) + "."
                            key = normalize_letters(text)
                            if key in seen:
                                continue
                            seen.add(key)
                            operation = {
                                "kind": "seed_half_mirrored_insertion_and_joint_resegmentation",
                                "seed_half_letters": len(SEED_TAPE) // 2,
                                "insertion_position": position,
                                "inserted_letter": letter,
                                "left_template": left_name,
                                "right_template": right_name,
                            }
                            row = _audit(text, operation, left, right)
                            row["diagnostic_bigram_score"] = round(_bigram_score(words), 4)
                            rows.append(row)
                            stats["exact_candidates"] += 1
    rows.sort(key=lambda row: (row["mechanically_admitted"], row["diagnostic_bigram_score"]), reverse=True)
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {
        "status": "complete_seed_boundary_shift_joint_resegmentation_no_promotion",
        "seed": SEED_TEXT,
        "seed_letters": len(SEED_TAPE),
        "config": {
            "target_lengths": "40 letters (one mirrored insertion into the 19-letter seed half)",
            "joint_content_replacement": "all eight seed content words excluded from output; at least two must change",
            "templates": [name for name, _ in TEMPLATES],
            "independent_ascii_audit": True,
            "central_anti_shortcut_checks": True,
            "wrapper_or_seed_interior_promotion": False,
        },
        "stats": dict(stats),
        "failure_samples": failures,
        "admitted": admitted,
        "near_misses": rows[:100],
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "role_inventory": "Brown majority universal POS with wordfreq >= 3.0; seed content excluded",
            "source_text_copied": False,
        },
        "next_operator": (
            "Retain the same typed bilateral residual state but permit two-character mirrored insertions and "
            "variable-length clause templates; carry lexical valency and punctuation-clause boundaries in the "
            "memo key, then independently exact-audit every N >= 39 closure before any reader gate."
        ),
        "reader_gate": "No output is human readability evidence; intact prose and shuffled controls remain required.",
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
