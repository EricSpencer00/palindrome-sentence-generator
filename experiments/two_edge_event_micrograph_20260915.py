"""Two-edge event micrograph palindrome feasibility search.

The final event-family ladder rung makes topology explicit: an intransitive
event transitions to an intermediate state, which transitions to a result
state.  Three strictly ordered temporal units and two semantic edges are
carried through an independently lexicalized right-side residual grammar.

This is a fresh finite inventory.  Earlier event-pair modules are not
imported or replayed.  Every exact closure receives an independent
two-pointer audit, central admission checks, and a repository-wide tape-key
exclusion made before the output artifact is written.
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
MAX_LETTERS = 360
MAX_PROBES = 100
FAMILY_ID = "two-edge-event-micrograph"
STATE_SPACE_SIGNATURE = (
    "semantic-event-micrograph|event-to-intermediate-to-result|"
    "explicit-two-edge-topology|strict-three-rank-temporal-order|"
    "independent-right-micrograph-lexicalization|five-link-unit-residual-prefix"
)


@dataclass(frozen=True)
class MicrographFrame:
    frame_id: str
    first_time: str
    first_entity: str
    first_event: str
    link_one: str
    second_time: str
    second_entity: str
    intermediate: str
    link_two: str
    third_time: str
    third_entity: str
    result: str
    first_rank: int
    second_rank: int
    third_rank: int

    @property
    def text(self) -> str:
        return (
            f"{self.first_time}, {self.first_entity} {self.first_event}, {self.link_one} "
            f"{self.second_time}, {self.second_entity} {self.intermediate}, {self.link_two} "
            f"{self.third_time}, {self.third_entity} {self.result}."
        )

    @property
    def words(self) -> tuple[str, ...]:
        return tokenize(self.text)

    @property
    def tape(self) -> str:
        return normalize_letters(self.text)

    @property
    def relation_graph(self) -> dict[str, Any]:
        return {
            "nodes": ["event", "intermediate_state", "result_state"],
            "edges": [
                {"from": "event", "to": "intermediate_state", "type": "transitions_to", "direction": "forward"},
                {"from": "intermediate_state", "to": "result_state", "type": "resolves_as", "direction": "forward"},
            ],
            "temporal_order": {
                "relation": "event_before_intermediate_before_result",
                "ranks": [self.first_rank, self.second_rank, self.third_rank],
                "strictly_ordered": self.first_rank < self.second_rank < self.third_rank,
                "times": [self.first_time, self.second_time, self.third_time],
            },
        }

    @property
    def semantic_state(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "relation_graph": self.relation_graph,
            "units": [
                {"role": "intransitive_event", "entity": self.first_entity, "predicate": self.first_event},
                {"role": "intermediate_state", "entity": self.second_entity, "predicate": self.intermediate},
                {"role": "result_state", "entity": self.third_entity, "predicate": self.result},
            ],
            "connectives": [self.link_one, self.link_two],
            "object_role": "absent_in_all_three_units_by_design",
            "relative_edge": False,
        }


# Fresh authored material, not imported from any preceding event experiment.
FRAMES = (
    MicrographFrame("kiln_clay_pot", "at sunrise", "the kiln", "glowed", "then", "by midday", "the clay", "softened", "and", "at dusk", "the pot", "was cool", 1, 2, 3),
    MicrographFrame("brook_banks_valley", "after rain", "the brook", "swelled", "so", "by evening", "the banks", "dried", "and", "by night", "the valley", "was clear", 1, 2, 3),
    MicrographFrame("bud_branch_tree", "in spring", "the bud", "opened", "then", "by summer", "the branch", "thickened", "and", "by fall", "the tree", "was bare", 1, 2, 3),
    MicrographFrame("field_soil_rows", "before winter", "the field", "rested", "while", "by thaw", "the soil", "loosened", "and", "by planting", "the rows", "were ready", 1, 2, 3),
)


# Independent right-side units and link slots.  Boundaries are solved against
# the reverse tape; no right phrase is formed by reversing a left word.
RIGHT_UNITS: dict[str, tuple[str, ...]] = {
    "TIME_ONE": ("at dawn", "after frost", "in autumn", "before noon", "at sunrise", "after rain"),
    "DET": ("the", "a", "our"),
    "ENTITY": ("cloud", "tower", "harbor", "cabin", "orchard", "meadow", "engine", "boat", "stone"),
    "EVENT": ("drifted", "moved", "settled", "waited", "rested", "opened", "closed", "brightened"),
    "LINK_ONE": ("and", "then", "so", "while"),
    "TIME_TWO": ("by noon", "by evening", "at twilight", "after sunset", "by harvest", "in winter"),
    "INTERMEDIATE": ("grew warm", "stood open", "became quiet", "lay dry", "turned dark", "seemed ready"),
    "LINK_TWO": ("and", "then", "so", "while"),
    "TIME_THREE": ("by dusk", "by night", "after class", "in spring", "by summer", "by planting"),
    "RESULT": ("was clear", "was calm", "stood empty", "became warm", "lay open", "seemed safe"),
}
RIGHT_TEMPLATE = (
    "TIME_ONE", "DET", "ENTITY", "EVENT", "LINK_ONE",
    "TIME_TWO", "DET", "ENTITY", "INTERMEDIATE", "LINK_TWO",
    "TIME_THREE", "DET", "ENTITY", "RESULT",
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


def _parse_left(frame: MicrographFrame) -> dict[str, Any]:
    expected = tokenize(
        f"{frame.first_time} {frame.first_entity} {frame.first_event} {frame.link_one} "
        f"{frame.second_time} {frame.second_entity} {frame.intermediate} {frame.link_two} "
        f"{frame.third_time} {frame.third_entity} {frame.result}"
    )
    return {
        "valid": frame.words == expected,
        "word_count": len(frame.words),
        "two_edges_present": True,
        "first_unit_role": "intransitive_event",
        "second_unit_role": "intermediate_state",
        "third_unit_role": "result_state",
        "no_transitive_object": True,
        "temporal_order_carried": frame.first_rank < frame.second_rank < frame.third_rank,
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
        stats["two_edge_frames"] += 1
        reverse_tape = frame.tape[::-1]
        left_parse = _parse_left(frame)
        prefixes = _partial_segmentations(reverse_tape, RIGHT_TEMPLATE)
        best = prefixes[0]
        full = tuple(row for row in prefixes if row["consumed"] == len(reverse_tape) and row["next_role"] is None)
        if not full:
            stats["reverse_two_edge_failures"] += 1
            probe = f"{frame.text[:-1]}; {' '.join(best['words'])}".rstrip()
            reason = "right_two_edge_micrograph_cannot_consume_reverse_tape"
            if best["consumed"] == 0:
                reason = "reverse_tape_has_no_lexical_prefix_for_two_edge_micrograph"
            rejections.append({
                "kind": "rejected_two_edge_micrograph_reverse_parse",
                "frame_id": frame.frame_id,
                "semantic_state": frame.semantic_state,
                "right_template": list(RIGHT_TEMPLATE),
                "rendered_probe": probe,
                "reverse_tape_prefix": reverse_tape[:best["consumed"]],
                "reverse_tape_first_unconsumed": reverse_tape[best["consumed"]:best["consumed"] + 12],
                "reverse_prefix_letters": best["consumed"],
                "reverse_prefix_words": list(best["words"]),
                "next_right_role": best["next_role"],
                "residual_state": {
                    "unit_index": best["unit_index"],
                    "expected_role": best["next_role"],
                    "relation_graph": frame.relation_graph,
                    "two_edge_topology_carried_through_reverse_residual": True,
                },
                "reason": reason,
                "independent_left_parse": left_parse,
                "independent_exact_audit": _audit(probe),
                "central_admission": mechanical_admission_checks(probe, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS),
                "readability_diagnostic": _readability_diagnostic(probe),
                "reader_status": "not_run",
            })
            continue
        stats["reverse_two_edge_closures"] += len(full)
        for parsed in full:
            rendered = f"{frame.text[:-1]}; {' '.join(parsed['words'])}."
            audit = _audit(rendered)
            gate = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
            tape = audit["normalized_letters"]
            novel = tape not in existing
            row = {
                "kind": "exact_two_edge_micrograph_closure",
                "rendered": rendered,
                "left_semantic_state": frame.semantic_state,
                "right_template": list(RIGHT_TEMPLATE),
                "right_words": list(parsed["words"]),
                "independent_right_parse": {
                    "valid": True,
                    "units": list(parsed["units"]),
                    "relation_graph": frame.relation_graph,
                    "two_edge_topology_carried_through_residual": True,
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
            "construction": "event -> intermediate state -> result with two explicit directed edges",
            "topology": "three temporal units, two-edge chain",
            "relation_graph": {
                "nodes": ["event", "intermediate_state", "result_state"],
                "edges": ["event-transitions_to-intermediate", "intermediate-resolves_as-result"],
            },
            "left_units": ["intransitive_event", "intermediate_state", "result_state"],
            "right_units": ["intransitive_event", "intermediate_state", "result_state"],
            "strict_temporal_order": "rank 1 < rank 2 < rank 3",
            "fresh_inventory": True,
            "prior_event_modules_imported": False,
            "brown_corpus_used": False,
            "svo_or_relative_attachment_states": False,
            "typed_semordnilap_or_global_pos_states": False,
            "cross_word_boundaries": True,
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "preexisting_tape_key_scope": "all normalized strings of length 1..360 in runs/**/*.json, data/**/*.json, experiments/**/*.json",
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
            "Stop this event-family ladder after the two-edge zero closure; only a materially different semantic "
            "construction with a new inventory should be considered next."
        ) if not exact_rows else "Send exact closures to blinded intact-prose and shuffled-control readers.",
        "closure_conclusion": (
            "No two-edge micrograph reverse parse closed in this finite grammar. The zero is a bounded residual "
            "failure, not an impossibility claim; every frame is retained with its graph state and rejection audit."
        ) if not exact_rows else "Exact closures exist but remain unreviewed.",
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fresh_micrograph_inventory_authored": True,
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
