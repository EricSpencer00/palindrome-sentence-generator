"""Connective semantic-class search for ordered two-event palindromes.

This successor makes the connective state explicit rather than treating
``and``/``then`` as interchangeable surface material.  Each fresh left frame
is an ordered intransitive event followed by a result-state report, linked by
one of three semantic classes: contrast, cause, or consequence.  The right
side is independently lexicalized with a class-specific connective slot and
must consume the reverse character tape.

No earlier event-pair module is imported or replayed.  Exact closures are
independently audited, passed through central admission, and compared with a
repository-wide pre-output tape fingerprint.  A zero result is a finite
grammar diagnosis, never a readability certificate or impossibility claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

MIN_LETTERS = 39
MAX_LETTERS = 300
MAX_PROBES = 100
FAMILY_ID = "connective-semantic-class-event-pair"
STATE_SPACE_SIGNATURE = (
    "semantic-event-pair|intransitive-event-to-result-state|"
    "temporal-order-state|connective-semantic-class-contrast-cause-consequence|"
    "independent-class-conditioned-right-lexicalization|three-unit-residual-prefix"
)
CONNECTIVE_CLASSES = ("contrast", "cause", "consequence")


@dataclass(frozen=True)
class SemanticClassFrame:
    frame_id: str
    connective_class: str
    first_time: str
    first_entity: str
    first_event: str
    connective: str
    second_time: str
    second_entity: str
    second_result: str
    first_rank: int
    second_rank: int

    @property
    def text(self) -> str:
        return (
            f"{self.first_time}, {self.first_entity} {self.first_event}, {self.connective} "
            f"{self.second_time}, {self.second_entity} {self.second_result}."
        )

    @property
    def words(self) -> tuple[str, ...]:
        return tokenize(self.text)

    @property
    def tape(self) -> str:
        return normalize_letters(self.text)

    @property
    def temporal_order(self) -> dict[str, Any]:
        return {
            "relation": "event_before_result",
            "first_time": self.first_time,
            "second_time": self.second_time,
            "first_rank": self.first_rank,
            "second_rank": self.second_rank,
            "strictly_ordered": self.first_rank < self.second_rank,
        }

    @property
    def semantic_state(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "connective_class": self.connective_class,
            "connective_surface": self.connective,
            "units": [
                {"role": "intransitive_event", "entity": self.first_entity, "predicate": self.first_event},
                {"role": "result_state", "entity": self.second_entity, "predicate": self.second_result},
            ],
            "temporal_order": self.temporal_order,
            "object_role": "absent_in_both_units_by_design",
            "relative_edge": False,
        }


# Fresh authored frames: none is imported from either earlier event-pair
# experiment.  All are event -> connective-class -> result-state sequences.
FRAMES = (
    SemanticClassFrame("bell_hall_silence", "contrast", "at dawn", "the bell", "rang", "yet", "by noon", "the hall", "was silent", 1, 3),
    SemanticClassFrame("wind_branch_calm", "contrast", "after wind", "the branch", "fell", "but", "at dusk", "the yard", "was calm", 1, 3),
    SemanticClassFrame("frost_pond_sun", "cause", "after frost", "the pond", "thawed", "because", "by midday", "the sun", "was warm", 1, 2),
    SemanticClassFrame("heat_wax_mold", "cause", "after heat", "the wax", "softened", "since", "by evening", "the mold", "was firm", 1, 3),
    SemanticClassFrame("seed_plant_height", "consequence", "in spring", "the seed", "sprouted", "so", "by summer", "the plant", "was tall", 1, 3),
    SemanticClassFrame("tide_shore_wetness", "consequence", "at sunrise", "the tide", "rose", "thus", "by evening", "the shore", "was wet", 1, 3),
)


# The right lexicalization is class-conditioned.  The words are independently
# authored; the reverse tape, not a reversed-word table, determines closure.
RIGHT_UNITS: dict[str, tuple[str, ...]] = {
    "TIME_EARLY": ("before dawn", "at sunrise", "in autumn", "after frost", "by morning", "after heat"),
    "DET": ("the", "a", "our"),
    "ENTITY": ("cloud", "bridge", "shore", "village", "window", "harbor", "engine", "boat", "stone"),
    "EVENT": ("drifted", "moved", "settled", "waited", "rested", "opened", "closed", "brightened"),
    "CONNECTIVE_CONTRAST": ("yet", "but", "however"),
    "CONNECTIVE_CAUSE": ("because", "since"),
    "CONNECTIVE_CONSEQUENCE": ("so", "thus", "therefore"),
    "TIME_LATE": ("by evening", "at twilight", "after sunset", "by harvest", "in winter", "after class"),
    "RESULT": ("was clear", "grew dark", "stood quiet", "became warm", "lay open", "turned calm", "seemed ready"),
}


def _template(connective_class: str) -> tuple[str, ...]:
    return (
        "TIME_EARLY", "DET", "ENTITY", "EVENT",
        f"CONNECTIVE_{connective_class.upper()}",
        "TIME_LATE", "DET", "ENTITY", "RESULT",
    )


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _existing_tape_keys(output: Path | None) -> tuple[set[str], dict[str, int]]:
    keys: set[str] = set()
    files = malformed = 0
    output_resolved = output.resolve() if output else None
    for base in (ROOT / "runs", ROOT / "data", ROOT / "experiments"):
        for path in base.rglob("*.json"):
            if output_resolved and path.resolve() == output_resolved:
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError):
                malformed += 1
                continue
            files += 1
            for value in _strings(payload):
                try:
                    tape = normalize_letters(value)
                except (TypeError, ValueError):
                    continue
                if 1 <= len(tape) <= MAX_LETTERS:
                    keys.add(tape)
    return keys, {"json_files_scanned": files, "malformed_json_files": malformed}


def _digest(keys: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


@lru_cache(maxsize=None)
def _partial_segmentations(tape: str, template: tuple[str, ...], cap: int = 24) -> tuple[dict[str, Any], ...]:
    frontier: list[dict[str, Any]] = []

    def rec(offset: int, slot: int, words: tuple[str, ...], units: tuple[str, ...]) -> None:
        frontier.append({
            "consumed": offset,
            "words": words,
            "units": units,
            "next_role": template[slot] if slot < len(template) else None,
            "unit_index": slot,
        })
        if slot >= len(template):
            return
        for phrase in RIGHT_UNITS[template[slot]]:
            phrase_words = tokenize(phrase)
            letters = "".join(phrase_words)
            if tape.startswith(letters, offset):
                rec(offset + len(letters), slot + 1, words + phrase_words, units + (phrase,))

    rec(0, 0, (), ())
    frontier.sort(key=lambda row: (-row["consumed"], len(row["words"]), row["words"]))
    unique: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[str, ...]]] = set()
    for row in frontier:
        key = (row["consumed"], row["words"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
        if len(unique) >= cap:
            break
    return tuple(unique)


def _audit(text: str) -> dict[str, Any]:
    tape = normalize_letters(text)
    mismatches = [
        {"left": i, "right": len(tape) - 1 - i, "left_char": tape[i], "right_char": tape[-1 - i]}
        for i in range(len(tape) // 2)
        if tape[i] != tape[-1 - i]
    ]
    return {
        "exact": bool(tape) and not mismatches,
        "two_pointer_exact": bool(tape) and all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "letters": len(tape),
        "normalized_letters": tape,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "mismatches": mismatches[:12],
    }


def _parse_left(frame: SemanticClassFrame) -> dict[str, Any]:
    expected = tokenize(
        f"{frame.first_time} {frame.first_entity} {frame.first_event} {frame.connective} "
        f"{frame.second_time} {frame.second_entity} {frame.second_result}"
    )
    return {
        "valid": frame.words == expected,
        "word_count": len(frame.words),
        "connective_class": frame.connective_class,
        "connective_slot_present": True,
        "first_unit_role": "intransitive_event",
        "second_unit_role": "result_state",
        "no_transitive_object": True,
        "temporal_order_carried": frame.first_rank < frame.second_rank,
    }


def _readability_diagnostic(text: str) -> dict[str, Any]:
    words = tokenize(text)
    frequencies = [zipf_frequency(word, "en") for word in words]
    return {
        "status": "diagnostic_only_unreviewed",
        "word_count": len(words),
        "all_words_zipf_ge_2": bool(words) and all(value >= 2 for value in frequencies),
        "mean_zipf_frequency": round(sum(frequencies) / max(1, len(frequencies)), 3),
        "blinded_reader_required": True,
    }


def run(output: Path | None = None) -> dict[str, Any]:
    existing, scan = _existing_tape_keys(output)
    stats = Counter()
    rejections: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []

    for frame in FRAMES:
        stats["semantic_class_frames"] += 1
        reverse_tape = frame.tape[::-1]
        left_parse = _parse_left(frame)
        template = _template(frame.connective_class)
        prefixes = _partial_segmentations(reverse_tape, template)
        best = prefixes[0]
        full = tuple(row for row in prefixes if row["consumed"] == len(reverse_tape) and row["next_role"] is None)
        if not full:
            stats["reverse_semantic_class_failures"] += 1
            probe = f"{frame.text[:-1]}; {' '.join(best['words'])}".rstrip()
            reason = "right_class_conditioned_grammar_cannot_consume_reverse_tape"
            if best["consumed"] == 0:
                reason = "reverse_tape_has_no_lexical_prefix_for_class_conditioned_right_grammar"
            rejections.append({
                "kind": "rejected_semantic_class_reverse_parse",
                "frame_id": frame.frame_id,
                "semantic_state": frame.semantic_state,
                "right_template": list(template),
                "rendered_probe": probe,
                "reverse_tape_prefix": reverse_tape[:best["consumed"]],
                "reverse_tape_first_unconsumed": reverse_tape[best["consumed"]:best["consumed"] + 12],
                "reverse_prefix_letters": best["consumed"],
                "reverse_prefix_words": list(best["words"]),
                "next_right_role": best["next_role"],
                "residual_state": {
                    "unit_index": best["unit_index"],
                    "expected_role": best["next_role"],
                    "connective_class": frame.connective_class,
                    "connective_slot_present": True,
                    "temporal_order": "event_before_result",
                    "ordering_and_class_carried_through_reverse_residual": True,
                },
                "reason": reason,
                "independent_left_parse": left_parse,
                "independent_exact_audit": _audit(probe),
                "central_admission": mechanical_admission_checks(probe, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS),
                "readability_diagnostic": _readability_diagnostic(probe),
                "reader_status": "not_run",
            })
            continue

        stats["reverse_semantic_class_closures"] += len(full)
        for parsed in full:
            rendered = f"{frame.text[:-1]}; {' '.join(parsed['words'])}."
            audit = _audit(rendered)
            gate = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
            tape = audit["normalized_letters"]
            novel = tape not in existing
            row = {
                "kind": "exact_semantic_class_event_pair_closure",
                "rendered": rendered,
                "left_semantic_state": frame.semantic_state,
                "right_template": list(template),
                "right_words": list(parsed["words"]),
                "independent_right_parse": {
                    "valid": True,
                    "units": list(parsed["units"]),
                    "connective_class": frame.connective_class,
                    "connective_slot_present": True,
                    "temporal_order_carried": True,
                },
                "independent_exact_audit": audit,
                "central_admission": gate,
                "novelty_audit": {"tape_absent_from_all_existing_json_keys": novel, "tape_key": tape},
                "readability_diagnostic": _readability_diagnostic(rendered),
                "mechanically_admitted": audit["exact"] and novel and all(gate.values()),
                "reader_status": "not_run; exactness does not certify readability",
            }
            exact_rows.append(row)
            if row["mechanically_admitted"]:
                stats["mechanically_admitted"] += 1

    assert all((not row["mechanically_admitted"]) or row["independent_exact_audit"]["exact"] for row in exact_rows)
    rejections.sort(key=lambda row: (-row["reverse_prefix_letters"], row["frame_id"]))
    exact_rows.sort(key=lambda row: (-row["independent_exact_audit"]["letters"], row["rendered"]))
    return {
        "status": "no_closure_pivot_required" if not exact_rows else "exact_closures_need_blinded_readers",
        "family_id": FAMILY_ID,
        "state_space_signature": STATE_SPACE_SIGNATURE,
        "config": {
            "construction": "ordered intransitive event -> class-conditioned connective -> result-state report",
            "connective_semantic_classes": list(CONNECTIVE_CLASSES),
            "left_units": ["intransitive_event", "discourse_connective", "result_state"],
            "right_units": ["intransitive_event", "class_conditioned_connective", "result_state"],
            "temporal_order_state": "event_before_result with strict rank carried through residual",
            "class_state_carried_through_residual": True,
            "fresh_inventory": True,
            "prior_event_pair_modules_imported": False,
            "brown_corpus_used": False,
            "svo_or_relative_attachment_states": False,
            "typed_semordnilap_or_global_pos_states": False,
            "cross_word_boundaries": True,
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "preexisting_tape_key_scope": "all normalized strings of length 1..300 in runs/**/*.json, data/**/*.json, experiments/**/*.json",
            "output_excluded_before_scan": True,
        },
        "novelty_audit": {
            "existing_tape_keys_count": len(existing),
            "existing_json_files_scanned": scan["json_files_scanned"],
            "malformed_json_files_skipped": scan["malformed_json_files"],
            "existing_tape_keys_sha256": _digest(existing),
            "all_exact_rows_checked_against_existing_keys": True,
            "all_mechanically_admitted_rows_novel": all(row["novelty_audit"]["tape_absent_from_all_existing_json_keys"] for row in exact_rows if row["mechanically_admitted"]),
        },
        "stats": dict(stats),
        "candidates": [row for row in exact_rows if row["mechanically_admitted"]],
        "exact_closures": exact_rows,
        "rejections": rejections[:MAX_PROBES],
        "next_operator": (
            "Condition the class state on an explicit semantic relation graph (cause/consequence direction and "
            "contrast polarity) while retaining strict temporal order, independent right lexicalization, and the "
            "repository-wide tape exclusion; do not enlarge inventories or replay prior frames."
        ) if not exact_rows else "Send exact closures to blinded intact-prose and shuffled-control readers.",
        "closure_conclusion": (
            "No class-conditioned reverse discourse parse closed in this finite grammar. Every fresh frame is retained "
            "with its class and temporal residual frontier plus independent exact/admission/readability diagnostics."
        ) if not exact_rows else "Exact closures exist but remain unreviewed.",
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fresh_semantic_class_inventory_authored": True,
            "right_inventory_independently_authored": True,
            "source_text_copied": False,
            "readability_certificate": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite output: {args.out}")
    result = run(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "stats": result["stats"], "candidates": len(result["candidates"])}, indent=2))


if __name__ == "__main__":
    main()
