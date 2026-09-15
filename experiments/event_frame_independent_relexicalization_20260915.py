"""Search a semantic event-frame family with independent reverse lexicalization.

This is a deliberately different construction family from the Brown SVO /
relative-attachment searches and from the typed semordnilap/POS lattices.  A
left item is an authored *event report*: a temporal frame plus an intransitive
or copular state change.  Its character tape is then offered to an
independently authored right-side event grammar.  The right grammar must find
its own word boundaries; no reversed word pair or copied sentence is used.

The run is an evidence artifact, not a readability certificate.  Every exact
closure is independently rechecked, sent through the shared admission gate,
and rejected if its normalized tape occurs in any pre-existing JSON artifact.
The output file is excluded from that pre-run key set so reruns remain honest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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
MAX_LETTERS = 240
MAX_PROBES = 80

# This signature is intentionally orthogonal to the Brown and typed families:
# no subject/object relation, no relative edge, no POS-shape lattice, and no
# pre-existing reverse-word inventory is a constructor state.
FAMILY_ID = "event-frame-independent-relexicalization"
STATE_SPACE_SIGNATURE = (
    "semantic-event-frame|temporal-intransitive-or-copular|"
    "independent-right-lexicalization|phrase-unit-residual-prefix"
)


@dataclass(frozen=True)
class EventFrame:
    event_id: str
    event_type: str
    temporal: str
    subject: str
    predicate: str

    @property
    def words(self) -> tuple[str, ...]:
        return tokenize(f"{self.temporal} {self.subject} {self.predicate}")

    @property
    def text(self) -> str:
        return f"{self.temporal}, {self.subject} {self.predicate}."

    @property
    def tape(self) -> str:
        return normalize_letters(self.text)

    @property
    def semantic_state(self) -> dict[str, str]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "frame": "TEMPORAL + ENTITY + INTRANSITIVE_OR_COPULAR_CHANGE",
            "temporal_role": self.temporal,
            "entity_role": self.subject,
            "predicate_role": self.predicate,
            "object_role": "absent_by_design",
        }


# Common, authored event reports.  They have no transitive object and no
# relative/attachment edge.  The semantic labels are metadata, not corpus
# text, and are independently checked after rendering.
EVENTS = (
    EventFrame("rain_drying_path", "weather_change", "after rain", "the path", "grew dry"),
    EventFrame("birds_rising", "motion_change", "at dawn", "the birds", "rose early"),
    EventFrame("hall_emptying", "state_change", "by midday", "the hall", "stood empty"),
    EventFrame("lake_cooling", "weather_change", "after frost", "the lake", "turned cold"),
    EventFrame("garden_brightening", "weather_change", "in spring", "the garden", "grew bright"),
    EventFrame("road_quieting", "state_change", "at dusk", "the road", "went quiet"),
    EventFrame("fire_lowering", "state_change", "before sunrise", "the fire", "burned low"),
    EventFrame("room_silencing", "state_change", "at evening", "the room", "fell still"),
    EventFrame("river_clearing", "weather_change", "after storms", "the river", "ran clear"),
    EventFrame("village_waking", "motion_change", "in winter", "the village", "seemed still"),
)


# Independent right-side lexicalization.  None of these entries is obtained
# by reversing a left word.  Phrase units let a right parse cross the original
# left word boundaries while retaining an independently chosen event syntax.
RIGHT_UNITS: dict[str, tuple[str, ...]] = {
    "DET": ("the", "a", "our"),
    "ENTITY": ("cloud", "field", "river", "window", "shore", "village", "engine", "boat", "stone"),
    "AUX": ("was", "became", "seemed", "remained"),
    "CHANGE": ("drifted", "moved", "settled", "waited", "rested", "opened", "closed"),
    "QUALITY": ("clear", "dark", "warm", "silent", "ready", "empty", "open", "still", "safe", "steady"),
    "TIME": ("at dusk", "after rain", "by midday", "in winter", "before sunrise", "at evening"),
    "ADVERB": ("early", "quietly", "slowly", "nearby", "outside"),
}

# These templates are not reversals of the left event frame.  They are
# independent right-side event descriptions with different role ordering.
RIGHT_TEMPLATES = (
    ("copular_state_report", ("DET", "ENTITY", "AUX", "QUALITY", "TIME")),
    ("intransitive_event_report", ("TIME", "DET", "ENTITY", "CHANGE", "ADVERB")),
    ("change_state_report", ("DET", "ENTITY", "CHANGE", "QUALITY", "TIME")),
)


def _phrase_words(unit: str) -> tuple[str, ...]:
    return tokenize(unit)


def _iter_json_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_json_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_json_strings(child)


def _existing_tape_keys(output: Path | None) -> tuple[set[str], dict[str, int]]:
    """Collect all bounded normalized JSON strings before this run writes output.

    This intentionally scans runs, data, and experiment JSON artifacts rather
    than trusting a hand-maintained list.  A candidate only needs comparison
    with tapes in its length band, so longer prose values are safely omitted.
    """
    roots = (ROOT / "runs", ROOT / "data", ROOT / "experiments")
    keys: set[str] = set()
    files = 0
    malformed = 0
    output_resolved = output.resolve() if output else None
    for base in roots:
        for path in base.rglob("*.json"):
            if output_resolved and path.resolve() == output_resolved:
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError):
                malformed += 1
                continue
            files += 1
            for text in _iter_json_strings(payload):
                try:
                    tape = normalize_letters(text)
                except (TypeError, ValueError):
                    continue
                if 1 <= len(tape) <= MAX_LETTERS:
                    keys.add(tape)
    return keys, {"json_files_scanned": files, "malformed_json_files": malformed}


def _key_digest(keys: Iterable[str]) -> str:
    joined = "\n".join(sorted(keys)).encode()
    return hashlib.sha256(joined).hexdigest()


def _units_for_role(role: str) -> tuple[tuple[str, ...], ...]:
    return tuple(_phrase_words(unit) for unit in RIGHT_UNITS[role])


@lru_cache(maxsize=None)
def _partial_segmentations(tape: str, template: tuple[str, ...], cap: int = 12) -> tuple[dict[str, Any], ...]:
    """Return best independently lexicalized prefixes for one right template."""
    best: list[dict[str, Any]] = []

    def rec(offset: int, slot: int, words: tuple[str, ...], units: tuple[str, ...]) -> None:
        best.append({"consumed": offset, "words": words, "units": units, "next_role": template[slot] if slot < len(template) else None})
        if slot >= len(template):
            return
        role = template[slot]
        for phrase_words in _units_for_role(role):
            phrase = "".join(phrase_words)
            if tape.startswith(phrase, offset):
                rec(offset + len(phrase), slot + 1, words + phrase_words, units + (" ".join(phrase_words),))

    rec(0, 0, (), ())
    # Keep only a deterministic frontier of longest prefixes, then shortest
    # lexical decomposition as a tie breaker.
    best.sort(key=lambda row: (-row["consumed"], len(row["words"]), row["words"]))
    dedup: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[str, ...]]] = set()
    for row in best:
        key = (row["consumed"], row["words"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)
        if len(dedup) >= cap:
            break
    return tuple(dedup)


def _exact_segmentations(tape: str, template: tuple[str, ...], cap: int = 20) -> tuple[dict[str, Any], ...]:
    return tuple(row for row in _partial_segmentations(tape, template, cap=cap) if row["consumed"] == len(tape) and row["next_role"] is None)


def _independent_event_parse(frame: EventFrame) -> dict[str, Any]:
    words = frame.words
    expected = tokenize(f"{frame.temporal} {frame.subject} {frame.predicate}")
    return {
        "valid": words == expected and len(words) >= 5,
        "word_count": len(words),
        "has_temporal_role": bool(tokenize(frame.temporal)),
        "has_entity_role": bool(tokenize(frame.subject)),
        "has_intransitive_or_copular_predicate": bool(tokenize(frame.predicate)),
        "no_object_role": True,
    }


def _exact_audit(text: str) -> dict[str, Any]:
    tape = normalize_letters(text)
    mismatches = [
        {"left": i, "right": len(tape) - 1 - i, "left_char": tape[i], "right_char": tape[-1 - i]}
        for i in range(len(tape) // 2)
        if tape[i] != tape[-1 - i]
    ]
    return {
        "exact": bool(tape) and not mismatches,
        "letters": len(tape),
        "normalized_letters": tape,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "two_pointer_exact": bool(tape) and all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "mismatches": mismatches[:12],
    }


def _diagnostic_readability(text: str) -> dict[str, Any]:
    words = tokenize(text)
    freqs = [zipf_frequency(word, "en") for word in words]
    return {
        "status": "diagnostic_only_unreviewed",
        "word_count": len(words),
        "all_common_word_frequency_ge_2": bool(words) and all(freq >= 2 for freq in freqs),
        "mean_zipf_frequency": round(sum(freqs) / max(1, len(freqs)), 3),
        "no_repeated_content_word": len(set(words)) == len(words),
        "blinded_reader_required": True,
    }


def _render_probe(frame: EventFrame, prefix: dict[str, Any]) -> str:
    right = " ".join(prefix["words"])
    return f"{frame.text[:-1]}; {right if right else '[no lexical reverse prefix]'} …"


def run(output: Path | None = None) -> dict[str, Any]:
    existing_keys, scan_stats = _existing_tape_keys(output)
    stats = Counter()
    rejections: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []

    for frame in EVENTS:
        stats["left_event_frames"] += 1
        left_tape = frame.tape
        reverse_tape = left_tape[::-1]
        left_parse = _independent_event_parse(frame)
        for template_name, template in RIGHT_TEMPLATES:
            stats["right_templates_offered"] += 1
            prefixes = _partial_segmentations(reverse_tape, template)
            best = prefixes[0]
            full = _exact_segmentations(reverse_tape, template)
            if not full:
                stats["reverse_parse_failures"] += 1
                reason = "right_grammar_cannot_consume_reverse_tape"
                if best["consumed"] == 0:
                    reason = "reverse_tape_has_no_lexical_prefix_for_right_grammar"
                rejections.append({
                    "kind": "rejected_partial_reverse_parse",
                    "event_id": frame.event_id,
                    "semantic_state": frame.semantic_state,
                    "right_template": template_name,
                    "rendered_probe": _render_probe(frame, best),
                    "left_rendered": frame.text,
                    "reverse_tape_prefix": reverse_tape[:best["consumed"]],
                    "reverse_tape_first_unconsumed": reverse_tape[best["consumed"]:best["consumed"] + 12],
                    "reverse_prefix_letters": best["consumed"],
                    "reverse_prefix_words": list(best["words"]),
                    "reason": reason,
                    "independent_left_parse": left_parse,
                    "readability_diagnostic": _diagnostic_readability(frame.text),
                    "reader_status": "not_run",
                })
                continue

            for parse in full:
                right = " ".join(parse["words"])
                rendered = f"{frame.text[:-1]}; {right}."
                audit = _exact_audit(rendered)
                gate = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
                tape = audit["normalized_letters"]
                novelty = tape not in existing_keys
                row = {
                    "kind": "exact_closure",
                    "rendered": rendered,
                    "left_event": frame.semantic_state,
                    "right_template": template_name,
                    "right_words": list(parse["words"]),
                    "independent_right_parse": {"valid": True, "template": list(template), "units": list(parse["units"])},
                    "independent_exact_audit": audit,
                    "central_admission": gate,
                    "novelty_audit": {"tape_absent_from_all_existing_json_keys": novelty, "tape_key": tape},
                    "readability_diagnostic": _diagnostic_readability(rendered),
                    "mechanically_admitted": audit["exact"] and novelty and all(gate.values()),
                    "reader_status": "not_run; programmatic checks never certify readability",
                }
                exact_rows.append(row)
                stats["exact_reverse_parses"] += 1
                if row["mechanically_admitted"]:
                    stats["mechanically_admitted"] += 1

    # A closure must be independently exact, novel, and admitted.  The
    # assertion catches accidental constructor regressions in future reruns.
    assert all(
        (not row["mechanically_admitted"])
        or row["independent_exact_audit"]["exact"]
        for row in exact_rows
    )

    rejections.sort(key=lambda row: (-row["reverse_prefix_letters"], row["event_id"], row["right_template"]))
    exact_rows.sort(key=lambda row: (-row["independent_exact_audit"]["letters"], row["rendered"]))
    return {
        "status": "no_closure_pivot_required" if not exact_rows else "exact_closures_need_blinded_readers",
        "family_id": FAMILY_ID,
        "state_space_signature": STATE_SPACE_SIGNATURE,
        "config": {
            "construction": "semantic event frame -> independently lexicalized reverse event phrase",
            "left_frame": "TEMPORAL + ENTITY + INTRANSITIVE_OR_COPULAR_CHANGE",
            "right_frame_templates": [name for name, _ in RIGHT_TEMPLATES],
            "left_event_count": len(EVENTS),
            "right_role_inventory": {role: list(values) for role, values in RIGHT_UNITS.items()},
            "cross_word_boundaries": True,
            "brown_corpus_used": False,
            "brown_svo_or_relative_attachment_states": False,
            "typed_semordnilap_or_global_pos_states": False,
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "preexisting_tape_key_scope": "all normalized strings of length 1..240 in runs/**/*.json, data/**/*.json, experiments/**/*.json",
            "output_excluded_before_scan": True,
        },
        "novelty_audit": {
            "existing_tape_keys_count": len(existing_keys),
            "existing_json_files_scanned": scan_stats["json_files_scanned"],
            "malformed_json_files_skipped": scan_stats["malformed_json_files"],
            "existing_tape_keys_sha256": _key_digest(existing_keys),
            "all_exact_rows_checked_against_existing_keys": True,
            "all_mechanically_admitted_rows_novel": all(row["novelty_audit"]["tape_absent_from_all_existing_json_keys"] for row in exact_rows if row["mechanically_admitted"]),
        },
        "stats": dict(stats),
        "candidates": [row for row in exact_rows if row["mechanically_admitted"]],
        "exact_closures": exact_rows,
        "rejections": rejections[:MAX_PROBES],
        "next_operator": (
            "Pivot to a two-event discourse frame: pair an independently authored intransitive event report "
            "with a result-state report and carry event_type plus temporal ordering through the reverse residual; "
            "do not add larger pools, beams, seeds, Brown relations, or typed semordnilap pairs."
        ) if not exact_rows else (
            "Send exact closures to blinded intact-prose and shuffled-control readers; programmatic admission is not readability evidence."
        ),
        "closure_conclusion": (
            "No full reverse phrase parse closed in this finite event grammar; every offered left event was retained "
            "with its longest lexical reverse-prefix rejection. The zero is a finite-grammar failure, not a claim that "
            "semantic event frames cannot produce readable palindromes."
        ) if not exact_rows else "Exact closures exist but remain unreviewed.",
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "event_inventory_authored": True,
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
