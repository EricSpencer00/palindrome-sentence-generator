"""Residual BFS over Brown-attested subject--verb--object relations.

Unlike isolated role-list searches, this operator first extracts compact SVO
relations actually attested in Brown.  It then expands each relation with
locally attested tense/number forms, builds an independent right-side residual
index, and intersects reversed character tapes.  Exact matches are still only
proposals: the shared anti-shortcut gate and a later blinded reader study are
required for promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS = 39
MAX_LETTERS = 120
SHORT = frozenset("a an the".split())


@dataclass(frozen=True)
class Relation:
    subject: str
    verb: str
    object: str
    subject_det: str = ""
    object_det: str = ""
    subject_number: str = "sing"
    object_number: str = "sing"
    source_sentence: str = ""

    @property
    def tape(self) -> str:
        return "".join(x for x in (self.subject_det, self.subject, self.verb, self.object_det, self.object))

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(x for x in (self.subject_det, self.subject, self.verb, self.object_det, self.object) if x)


def _word_ok(word: str) -> bool:
    return word.isascii() and word.isalpha() and 2 <= len(word) <= 12 and zipf_frequency(word, "en") >= 3.0


def _extract_relations(limit: int = 80_000) -> tuple[tuple[Relation, ...], dict[str, int]]:
    """Extract one-word-NP / verb / one-word-NP relations from Brown."""
    from nltk.corpus import brown

    rows: dict[tuple[str, str, str, str, str], Relation] = {}
    sentence_count = 0
    for tagged in brown.tagged_sents(tagset="universal"):
        sentence_count += 1
        words = [word.casefold() for word, _ in tagged]
        tags = [tag for _, tag in tagged]
        for i in range(len(words) - 4):
            # DET? NOUN VERB DET? NOUN.  We require lexical verbs and nouns;
            # adjective-rich relations are added by the next operator.
            det_a = words[i] if tags[i] == "DET" and words[i] in SHORT else ""
            subj_i = i + 1 if det_a else i
            if tags[subj_i] not in {"NOUN", "PROPN"}:
                continue
            verb_i = subj_i + 1
            if tags[verb_i] not in {"VERB", "AUX"} or not _word_ok(words[verb_i]):
                continue
            det_b_i = verb_i + 1
            det_b = words[det_b_i] if tags[det_b_i] == "DET" and words[det_b_i] in SHORT else ""
            obj_i = det_b_i + 1 if det_b else det_b_i
            if obj_i >= len(words) or tags[obj_i] not in {"NOUN", "PROPN"}:
                continue
            if not (_word_ok(words[subj_i]) and _word_ok(words[obj_i])):
                continue
            # Brown's universal tag retains number poorly; suffix provides a
            # reproducible conservative feature used by the variant expander.
            key = (words[subj_i], words[verb_i], words[obj_i], det_a, det_b)
            rows.setdefault(
                key,
                Relation(
                    subject=words[subj_i], verb=words[verb_i], object=words[obj_i],
                    subject_det=det_a, object_det=det_b,
                    subject_number="plur" if words[subj_i].endswith("s") else "sing",
                    object_number="plur" if words[obj_i].endswith("s") else "sing",
                    source_sentence=" ".join(words),
                ),
            )
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    return tuple(rows.values()), {"brown_sentences_scanned": sentence_count, "attested_relations": len(rows)}


def _attested_inflections(relations: tuple[Relation, ...]) -> dict[str, tuple[str, ...]]:
    """Collect forms observed in Brown, keyed by a relation lexeme."""
    forms: dict[str, set[str]] = defaultdict(set)
    for row in relations:
        for word, role in ((row.subject, "N"), (row.verb, "V"), (row.object, "N")):
            forms[word].add(word)
            # The derived forms are admitted only if they are attested among
            # the extracted Brown relation vocabulary, below.
            if role == "N":
                forms[word].update({word + "s", word.removesuffix("s")})
            else:
                forms[word].update({word + "s", word + "ed", word + "ing"})
    observed = {word for row in relations for word in (row.subject, row.verb, row.object)}
    return {key: tuple(sorted(value & observed)) for key, value in forms.items()}


def _variants(row: Relation, forms: dict[str, tuple[str, ...]], cap: int = 12):
    subjects = forms.get(row.subject, (row.subject,))
    verbs = forms.get(row.verb, (row.verb,))
    objects = forms.get(row.object, (row.object,))
    count = 0
    for subject in subjects:
        for verb in verbs:
            for obj in objects:
                # Preserve agreement where the observed frame permits a clear
                # singular/plural cue; this is a variant, not a new relation.
                if row.subject_number == "sing" and subject.endswith("s") and subject not in {"is", "was"}:
                    continue
                variant = Relation(subject, verb, obj, row.subject_det, row.object_det,
                                   row.subject_number, row.object_number, row.source_sentence)
                yield variant
                count += 1
                if count >= cap:
                    return


def _word_mirror(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return tuple(word[::-1] for word in reversed(left)) == right


def _audit(left: Relation, right: Relation, variant_left: Relation, variant_right: Relation) -> dict:
    text = " ".join(variant_left.words).capitalize() + "; " + " ".join(variant_right.words) + "."
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_letters": tape,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "independent_ascii_exact": bool(tape) and tape == tape[::-1],
        "independent_two_pointer": all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "left_relation": left.__dict__,
        "right_relation": right.__dict__,
        "left_variant": variant_left.__dict__,
        "right_variant": variant_right.__dict__,
        "word_order_shortcut": _word_mirror(variant_left.words, variant_right.words),
        "central_admission": checks,
        "mechanically_admitted": all(checks.values()) and not _word_mirror(variant_left.words, variant_right.words),
        "reader_status": "not_run; corpus attestation is not human readability evidence",
    }


def run() -> dict:
    relations, extraction_stats = _extract_relations()
    forms = _attested_inflections(relations)
    # Build the right residual index from independently selected relation rows.
    right_index: dict[str, list[tuple[int, Relation, Relation]]] = defaultdict(list)
    variant_count = 0
    for index, row in enumerate(relations):
        for variant in _variants(row, forms):
            tape = variant.tape
            if 15 <= len(tape) <= 60:
                right_index[tape].append((index, row, variant))
                variant_count += 1
    candidates: list[dict] = []
    seen: set[str] = set()
    stats = Counter(extraction_stats)
    stats["right_variants_indexed"] = variant_count
    for left_index, left in enumerate(relations):
        for left_variant in _variants(left, forms):
            reverse_tape = left_variant.tape[::-1]
            matches = right_index.get(reverse_tape, ())
            stats["left_variants_considered"] += 1
            if not matches:
                stats["residual_misses"] += 1
                continue
            for right_index_num, right, right_variant in matches:
                stats["residual_matches"] += 1
                if left_index == right_index_num:
                    stats["same_relation_rejections"] += 1
                    continue
                if len(left_variant.tape) + len(right_variant.tape) < MIN_LETTERS:
                    stats["short_rejections"] += 1
                    continue
                if set(left_variant.words + right_variant.words) & SEED_CONTENT:
                    stats["seed_overlap_rejections"] += 1
                    continue
                if _word_mirror(left_variant.words, right_variant.words):
                    stats["word_order_rejections"] += 1
                    continue
                text_key = left_variant.tape + right_variant.tape
                if text_key in seen:
                    continue
                seen.add(text_key)
                row = _audit(left, right, left_variant, right_variant)
                candidates.append(row)
                stats["candidates"] += 1
                if row["mechanically_admitted"]:
                    stats["mechanically_admitted"] += 1
    candidates.sort(key=lambda row: (row["mechanically_admitted"], row["letters"]), reverse=True)
    return {
        "status": "complete_brown_attested_relation_residual_bfs_no_reader_promotion",
        "config": {
            "relation_source": "NLTK Brown universal-tagged sentences",
            "relation_shape": "DET? NOUN VERB DET? NOUN",
            "inflection_policy": "only forms observed among extracted Brown relations; conservative agreement cue",
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "independent_right_residual_index": True,
            "catalogue_and_repetition_gate": True,
        },
        "stats": dict(stats),
        "admitted": [row for row in candidates if row["mechanically_admitted"]],
        "near_misses": candidates[:100],
        "residual_frontier": {
            "indexed_relation_tapes": len(right_index),
            "unmatched_left_variants": stats["residual_misses"],
            "matched_before_gates": stats["residual_matches"],
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_text_copied": False,
            "relation_rows_are_attested": True,
        },
        "next_operator": (
            "Keep the attested residual index but add two-adjacent-clause Brown frames with adjective and "
            "prepositional attachments; carry dependency/number/tense features through the memo key and audit "
            "each exact closure independently before human study."
        ),
        "reader_gate": "No corpus or programmatic result certifies readability; use intact prose and shuffled controls.",
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
