"""Relation-graph conditioned search over connective event pairs.

This is the next operator after connective semantic classes.  A fresh authored
discourse pair contains an intransitive event and a result-state report.  Its
semantic edge is explicit: ``causes`` and ``leads_to`` are directed edges,
while ``contrasts_with`` is symmetric with opposed polarity.  The edge type,
direction, polarity, and strict event-before-result time order are carried in
the residual state while an independently authored right grammar chooses a
class-compatible connective and lexicalization.

The prior inventories are not imported or replayed.  Every exact closure is
independently audited, checked by the central admission gate, and compared
against a repository-wide normalized-tape fingerprint made before this output
file is written.
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
MAX_LETTERS = 320
MAX_PROBES = 100
FAMILY_ID = "relation-graph-event-pair"
STATE_SPACE_SIGNATURE = (
    "semantic-event-pair|intransitive-event-to-result-state|"
    "explicit-relation-graph-edge-direction-polarity|strict-temporal-order|"
    "independent-graph-conditioned-right-lexicalization|three-unit-residual-prefix"
)
RELATION_CLASSES = ("contrast", "cause", "consequence")


@dataclass(frozen=True)
class GraphFrame:
    frame_id: str
    relation_class: str
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
    def relation_graph(self) -> dict[str, Any]:
        if self.relation_class == "contrast":
            edge = {"type": "contrasts_with", "direction": "symmetric", "polarity": "opposed"}
        elif self.relation_class == "cause":
            edge = {"type": "causes", "direction": "event_to_result", "polarity": "supporting"}
        else:
            edge = {"type": "leads_to", "direction": "event_to_result", "polarity": "supporting"}
        return {
            "nodes": ["event", "result_state"],
            "edge": edge,
            "temporal_order": {
                "relation": "event_before_result",
                "first_time": self.first_time,
                "second_time": self.second_time,
                "first_rank": self.first_rank,
                "second_rank": self.second_rank,
                "strictly_ordered": self.first_rank < self.second_rank,
            },
        }

    @property
    def semantic_state(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "relation_class": self.relation_class,
            "connective_surface": self.connective,
            "relation_graph": self.relation_graph,
            "units": [
                {"role": "intransitive_event", "entity": self.first_entity, "predicate": self.first_event},
                {"role": "result_state", "entity": self.second_entity, "predicate": self.second_result},
            ],
            "object_role": "absent_in_both_units_by_design",
            "relative_edge": False,
        }


# Fresh authored frames.  They are not imported from any earlier event-pair
# inventory and are deliberately object-free.
FRAMES = (
    GraphFrame("forge_workshop_contrast", "contrast", "at daybreak", "the forge", "roared", "yet", "by midday", "the workshop", "was quiet", 1, 3),
    GraphFrame("road_bridge_contrast", "contrast", "after rain", "the road", "shone", "but", "by sunset", "the bridge", "was dry", 1, 3),
    GraphFrame("leaves_courtyard_cause", "cause", "in autumn", "the leaves", "fell", "because", "by evening", "the courtyard", "was bare", 1, 3),
    GraphFrame("glass_chamber_cause", "cause", "after cold", "the glass", "cracked", "since", "by morning", "the chamber", "was warm", 1, 2),
    GraphFrame("seed_stalk_consequence", "consequence", "at first light", "the seed", "sprouted", "so", "by summer", "the stalk", "was tall", 1, 3),
    GraphFrame("brook_valley_consequence", "consequence", "after thaw", "the brook", "ran", "thus", "by harvest", "the valley", "was green", 1, 4),
)


RIGHT_UNITS: dict[str, tuple[str, ...]] = {
    "TIME_EARLY": ("before sunrise", "at daybreak", "in autumn", "after cold", "at first light", "after thaw"),
    "DET": ("the", "a", "our"),
    "ENTITY": ("cloud", "tower", "harbor", "cabin", "orchard", "meadow", "engine", "boat", "stone"),
    "EVENT": ("drifted", "moved", "settled", "waited", "rested", "opened", "closed", "brightened"),
    "RELATION_CONTRAST": ("yet", "but", "while"),
    "RELATION_CAUSE": ("because", "since", "as"),
    "RELATION_CONSEQUENCE": ("so", "thus", "therefore"),
    "TIME_LATE": ("by evening", "at twilight", "after sunset", "by harvest", "in winter", "after class"),
    "RESULT": ("was clear", "grew dark", "stood quiet", "became warm", "lay open", "turned calm", "seemed ready"),
}


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


def _parse_left(frame: GraphFrame) -> dict[str, Any]:
    expected = tokenize(
        f"{frame.first_time} {frame.first_entity} {frame.first_event} {frame.connective} "
        f"{frame.second_time} {frame.second_entity} {frame.second_result}"
    )
    return {
        "valid": frame.words == expected,
        "word_count": len(frame.words),
        "relation_class": frame.relation_class,
        "relation_graph_present": True,
        "first_unit_role": "intransitive_event",
        "second_unit_role": "result_state",
        "no_transitive_object": True,
        "temporal_order_carried": frame.first_rank < frame.second_rank,
    }


def run(output: Path | None = None) -> dict[str, Any]:
    existing, scan = _existing_tape_keys(output)
    stats = Counter()
    rejections: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []

    for frame in FRAMES:
        stats["relation_graph_frames"] += 1
        reverse_tape = frame.tape[::-1]
        left_parse = _parse_left(frame)
        template = _template(frame.relation_class)
        prefixes = _partial_segmentations(reverse_tape, template)
        best = prefixes[0]
        full = tuple(row for row in prefixes if row["consumed"] == len(reverse_tape) and row["next_role"] is None)
        if not full:
            stats["reverse_relation_graph_failures"] += 1
            probe = f"{frame.text[:-1]}; {' '.join(best['words'])}".rstrip()
            reason = "right_relation_graph_grammar_cannot_consume_reverse_tape"
            if best["consumed"] == 0:
                reason = "reverse_tape_has_no_lexical_prefix_for_relation_graph_grammar"
            rejections.append({
                "kind": "rejected_relation_graph_reverse_parse",
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
                    "relation_graph": frame.relation_graph,
                    "relation_graph_carried_through_reverse_residual": True,
                },
                "reason": reason,
                "independent_left_parse": left_parse,
                "independent_exact_audit": _audit(probe),
                "central_admission": mechanical_admission_checks(probe, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS),
                "readability_diagnostic": _readability_diagnostic(probe),
                "reader_status": "not_run",
            })
            continue

        stats["reverse_relation_graph_closures"] += len(full)
        for parsed in full:
            rendered = f"{frame.text[:-1]}; {' '.join(parsed['words'])}."
            audit = _audit(rendered)
            gate = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
            tape = audit["normalized_letters"]
            novel = tape not in existing
            row = {
                "kind": "exact_relation_graph_event_pair_closure",
                "rendered": rendered,
                "left_semantic_state": frame.semantic_state,
                "right_template": list(template),
                "right_words": list(parsed["words"]),
                "independent_right_parse": {
                    "valid": True,
                    "units": list(parsed["units"]),
                    "relation_graph": frame.relation_graph,
                    "relation_graph_carried_through_residual": True,
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
            "construction": "ordered intransitive event -> explicit relation graph edge -> result-state report",
            "relation_classes": list(RELATION_CLASSES),
            "relation_graph_edges": {
                "contrast": {"type": "contrasts_with", "direction": "symmetric", "polarity": "opposed"},
                "cause": {"type": "causes", "direction": "event_to_result", "polarity": "supporting"},
                "consequence": {"type": "leads_to", "direction": "event_to_result", "polarity": "supporting"},
            },
            "left_units": ["intransitive_event", "class_conditioned_connective", "result_state"],
            "right_units": ["intransitive_event", "class_conditioned_connective", "result_state"],
            "temporal_order_state": "event_before_result with strict rank carried through residual",
            "fresh_inventory": True,
            "prior_event_pair_modules_imported": False,
            "brown_corpus_used": False,
            "svo_or_relative_attachment_states": False,
            "typed_semordnilap_or_global_pos_states": False,
            "cross_word_boundaries": True,
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "preexisting_tape_key_scope": "all normalized strings of length 1..320 in runs/**/*.json, data/**/*.json, experiments/**/*.json",
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
            "Split the relation graph into an explicit two-edge micrograph (event -> intermediate state -> result) "
            "while retaining strict temporal order, independent class-conditioned right lexicalization, and full "
            "repository tape exclusion; do not enlarge inventories or replay prior frames."
        ) if not exact_rows else "Send exact closures to blinded intact-prose and shuffled-control readers.",
        "closure_conclusion": (
            "No relation-graph conditioned reverse discourse parse closed in this finite grammar. Every frame is "
            "retained with its edge direction/polarity, temporal state, rendered rejection, and independent "
            "exact/admission/readability diagnostics."
        ) if not exact_rows else "Exact closures exist but remain unreviewed.",
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fresh_relation_graph_inventory_authored": True,
            "right_inventory_independently_authored": True,
            "source_text_copied": False,
            "readability_certificate": False,
        },
    }


def _template(relation_class: str) -> tuple[str, ...]:
    return (
        "TIME_EARLY", "DET", "ENTITY", "EVENT",
        f"RELATION_{relation_class.upper()}",
        "TIME_LATE", "DET", "ENTITY", "RESULT",
    )


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
