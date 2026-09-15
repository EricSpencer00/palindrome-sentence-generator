"""Two-event discourse-frame palindrome constructor.

The pivot from the single-event experiment is explicit: each left proposal is
an ordered pair of event units, an intransitive event followed by its
result-state report.  A separate right grammar independently lexicalizes the
same two-unit discourse shape and attempts to consume the reverse character
tape.  Temporal ordering is a state variable, not a prose comment: the
constructor and the independent right parser both carry ``event_before_result``
and reject a reversed ordering.

This artifact intentionally has its own authored inventory and does not import
or rerun the single-event event-frame search.  It is a bounded feasibility run;
programmatic admission and exactness never certify human readability.
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
MAX_LETTERS = 260
MAX_PROBES = 100
FAMILY_ID = "two-event-discourse-frame"
STATE_SPACE_SIGNATURE = (
    "semantic-event-pair|intransitive-event-to-result-state|"
    "temporal-order-state|independent-right-discourse-lexicalization|"
    "two-unit-residual-prefix"
)


@dataclass(frozen=True)
class DiscourseFrame:
    frame_id: str
    event_type: str
    result_type: str
    first_time: str
    first_entity: str
    first_event: str
    second_time: str
    second_entity: str
    second_result: str
    first_rank: int
    second_rank: int

    @property
    def words(self) -> tuple[str, ...]:
        return tokenize(self.text)

    @property
    def text(self) -> str:
        return (
            f"{self.first_time}, {self.first_entity} {self.first_event}; "
            f"{self.second_time}, {self.second_entity} {self.second_result}."
        )

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
            "event_type": self.event_type,
            "result_type": self.result_type,
            "units": [
                {"role": "intransitive_event", "entity": self.first_entity, "predicate": self.first_event},
                {"role": "result_state", "entity": self.second_entity, "predicate": self.second_result},
            ],
            "temporal_order": self.temporal_order,
            "object_role": "absent_in_both_units_by_design",
            "relative_edge": False,
        }


# Fresh, authored discourse pairs.  This inventory is intentionally not
# imported from the preceding single-event run.  Neither unit has a transitive
# object or a relative-clause attachment.
DISCOURSE_FRAMES = (
    DiscourseFrame("showers_trail_ground", "weather", "visibility", "after showers", "the trail", "dried", "by midday", "the ground", "cleared", 1, 3),
    DiscourseFrame("sunrise_flock_meadow", "motion", "occupancy", "at sunrise", "the flock", "rose", "by morning", "the meadow", "emptied", 1, 2),
    DiscourseFrame("wind_leaves_yard", "motion", "rest", "after wind", "the leaves", "fell", "at twilight", "the yard", "rested", 1, 3),
    DiscourseFrame("winter_stream_bank", "weather", "temperature", "in winter", "the stream", "froze", "by spring", "the bank", "warmed", 1, 2),
    DiscourseFrame("sunset_lamps_street", "light", "visibility", "before sunset", "the lamps", "glowed", "at night", "the street", "dimmed", 1, 2),
    DiscourseFrame("thaw_soil_garden", "weather", "opening", "after thaw", "the soil", "softened", "by evening", "the garden", "opened", 1, 3),
    DiscourseFrame("drought_wells_fields", "weather", "condition", "after drought", "the wells", "lowered", "by harvest", "the fields", "hardened", 1, 4),
    DiscourseFrame("morning_bells_school", "sound", "quiet", "in morning", "the bells", "rang", "after class", "the school", "quieted", 1, 3),
)


# Independent right-side lexicalization.  Phrase units are selected without
# reversing left words; the only connection is the reverse tape constraint.
RIGHT_UNITS: dict[str, tuple[str, ...]] = {
    "TIME_EARLY": ("after storms", "at sunrise", "in autumn", "before noon", "after frost", "by morning"),
    "TIME_LATE": ("by evening", "at twilight", "after sunset", "by harvest", "in winter", "after class"),
    "DET": ("the", "a", "our"),
    "ENTITY": ("cloud", "bridge", "shore", "village", "window", "harbor", "engine", "boat", "stone"),
    "EVENT": ("drifted", "moved", "settled", "waited", "rested", "opened", "closed", "brightened"),
    "RESULT": ("was clear", "grew dark", "stood quiet", "became warm", "lay open", "turned calm", "seemed ready"),
}

RIGHT_TEMPLATE = (
    "TIME_EARLY", "DET", "ENTITY", "EVENT",
    "TIME_LATE", "DET", "ENTITY", "RESULT",
)


def _phrase_words(unit: str) -> tuple[str, ...]:
    return tokenize(unit)


@lru_cache(maxsize=None)
def _partial_segmentations(tape: str, template: tuple[str, ...], cap: int = 20) -> tuple[dict[str, Any], ...]:
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
        role = template[slot]
        for phrase in RIGHT_UNITS[role]:
            phrase_words = _phrase_words(phrase)
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


def _exact_segmentations(tape: str) -> tuple[dict[str, Any], ...]:
    return tuple(
        row for row in _partial_segmentations(tape, RIGHT_TEMPLATE)
        if row["consumed"] == len(tape) and row["next_role"] is None
    )


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


def _parse_left(frame: DiscourseFrame) -> dict[str, Any]:
    words = frame.words
    expected = tokenize(
        f"{frame.first_time} {frame.first_entity} {frame.first_event} "
        f"{frame.second_time} {frame.second_entity} {frame.second_result}"
    )
    return {
        "valid": words == expected,
        "word_count": len(words),
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


def _existing_tape_keys(output: Path | None) -> tuple[set[str], dict[str, int]]:
    """Fingerprint every bounded normalized JSON string before output exists."""
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


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _digest(keys: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def run(output: Path | None = None) -> dict[str, Any]:
    existing, scan = _existing_tape_keys(output)
    stats = Counter()
    rejections: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []

    for frame in DISCOURSE_FRAMES:
        stats["discourse_frames"] += 1
        left_tape = frame.tape
        reverse_tape = left_tape[::-1]
        left_parse = _parse_left(frame)
        prefixes = _partial_segmentations(reverse_tape, RIGHT_TEMPLATE)
        # Only retain the longest frontier for the rejection ledger; all
        # complete parses are handled below.  This keeps the artifact
        # inspectable while preserving exhaustive finite-grammar counts.
        best = prefixes[0]
        full = _exact_segmentations(reverse_tape)
        if not full:
            stats["reverse_discourse_failures"] += 1
            probe = f"{frame.text[:-1]}; {' '.join(best['words'])}".rstrip()
            reason = "right_discourse_grammar_cannot_consume_reverse_tape"
            if best["consumed"] == 0:
                reason = "reverse_tape_has_no_lexical_prefix_for_two_event_right_grammar"
            rejections.append({
                "kind": "rejected_two_event_reverse_parse",
                "frame_id": frame.frame_id,
                "semantic_state": frame.semantic_state,
                "rendered_probe": probe,
                "reverse_tape_prefix": reverse_tape[:best["consumed"]],
                "reverse_tape_first_unconsumed": reverse_tape[best["consumed"]:best["consumed"] + 12],
                "reverse_prefix_letters": best["consumed"],
                "reverse_prefix_words": list(best["words"]),
                "next_right_role": best["next_role"],
                "residual_state": {
                    "unit_index": best["unit_index"],
                    "expected_role": best["next_role"],
                    "temporal_order": "event_before_result",
                    "ordering_carried_through_reverse_residual": True,
                },
                "reason": reason,
                "independent_left_parse": left_parse,
                "independent_exact_audit": _audit(probe),
                "central_admission": mechanical_admission_checks(probe, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS),
                "readability_diagnostic": _readability_diagnostic(probe),
                "reader_status": "not_run",
            })
            continue

        stats["reverse_discourse_closures"] += len(full)
        for parsed in full:
            right = " ".join(parsed["words"])
            rendered = f"{frame.text[:-1]}; {right}."
            audit = _audit(rendered)
            gate = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
            tape = audit["normalized_letters"]
            novel = tape not in existing
            row = {
                "kind": "exact_two_event_closure",
                "rendered": rendered,
                "left_semantic_state": frame.semantic_state,
                "right_temporal_order": "event_before_result",
                "right_template": list(RIGHT_TEMPLATE),
                "right_words": list(parsed["words"]),
                "independent_right_parse": {"valid": True, "units": list(parsed["units"]), "temporal_order_carried": True},
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
            "construction": "ordered intransitive event report -> result-state report, independently lexicalized in reverse",
            "left_units": ["intransitive_event", "result_state"],
            "right_units": ["intransitive_event", "result_state"],
            "temporal_order_state": "event_before_result with strict rank carried in both parses",
            "discourse_frame_count": len(DISCOURSE_FRAMES),
            "right_template": list(RIGHT_TEMPLATE),
            "fresh_inventory": True,
            "single_event_inventory_imported": False,
            "brown_corpus_used": False,
            "svo_or_relative_attachment_states": False,
            "typed_semordnilap_or_global_pos_states": False,
            "cross_word_boundaries": True,
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "preexisting_tape_key_scope": "all normalized strings of length 1..260 in runs/**/*.json, data/**/*.json, experiments/**/*.json",
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
            "Add one independently lexicalized discourse connective between the two event units while retaining "
            "the strict event_before_result rank state and the repository-wide tape exclusion; do not enlarge either "
            "event inventory or fall back to Brown/POS/semordnilap search."
        ) if not exact_rows else "Send exact closures to blinded intact-prose and shuffled-control readers.",
        "closure_conclusion": (
            "No two-event reverse discourse parse closed in this finite grammar. The result is a finite residual "
            "failure, not evidence against semantic event pairing in general; each frame is retained with its exact "
            "reverse-prefix frontier and independent rejection checks."
        ) if not exact_rows else "Exact closures exist but remain unreviewed.",
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fresh_discourse_inventory_authored": True,
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
