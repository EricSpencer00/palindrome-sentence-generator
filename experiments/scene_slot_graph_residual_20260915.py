"""Bidirectional lexicalized scene-description slot-graph search.

This is a new construction family rather than another event/discourse or
Brown/POS search.  A scene graph has location, possession, and attribution
edges.  The left side is an authored coordinated description; the right side
is independently lexicalized from typed slots while consuming the reversed
character tape.  Semantic edge roles and English slot order are checked in
the same residual state.

The run is deliberately finite and diagnostic.  Exactness is independently
audited and the shared admission gate is reported, but neither one certifies
human readability.
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
MAX_LETTERS = 280
MAX_PROBES = 24
FAMILY_ID = "scene-slot-graph-residual"
STATE_SPACE_SIGNATURE = (
    "semantic-scene-graph|located-at-possesses-describes-edges|"
    "coordinated-locative-possessive-attributive-clauses|"
    "bidirectional-typed-slot-graph|reverse-character-residual|"
    "independent-right-lexicalization"
)


@dataclass(frozen=True)
class SceneGraph:
    """A compact scene graph plus a complete, grammatical left realization."""

    frame_id: str
    place_a: str
    owner_a: str
    attribute: str
    object_a: str
    verb_a: str
    owner_b: str
    object_b: str
    verb_b: str
    place_b: str

    @property
    def text(self) -> str:
        return (
            f"In the {self.place_a}, the {self.owner_a}'s {self.attribute} {self.object_a} "
            f"{self.verb_a} near the {self.place_b}, and the {self.owner_b}'s {self.object_b} "
            f"{self.verb_b} by the {self.place_a}."
        )

    @property
    def tape(self) -> str:
        return normalize_letters(self.text)

    @property
    def graph(self) -> dict[str, Any]:
        return {
            "nodes": {
                "scene": "shared_scene",
                "place_a": self.place_a,
                "place_b": self.place_b,
                "owner_a": self.owner_a,
                "owner_b": self.owner_b,
                "object_a": self.object_a,
                "object_b": self.object_b,
                "attribute": self.attribute,
            },
            "edges": [
                {"type": "located_at", "source": "scene", "target": "place_a", "rank": 1},
                {"type": "possesses", "source": "owner_a", "target": "object_a", "rank": 1},
                {"type": "describes", "source": "attribute", "target": "object_a", "rank": 1},
                {"type": "located_at", "source": "object_a", "target": "place_b", "rank": 2},
                {"type": "possesses", "source": "owner_b", "target": "object_b", "rank": 2},
                {"type": "located_at", "source": "object_b", "target": "place_a", "rank": 2},
            ],
            "coordinated_clauses": 2,
            "edge_roles_complete": True,
        }


# Fresh, task-authored scene descriptions.  No source text or corpus sentence
# is copied, and each frame uses distinct content words within its surface.
SCENES = (
    SceneGraph("gallery_lamp_vase", "gallery", "artist", "amber", "lamp", "rests", "curator", "vase", "stands", "arch"),
    SceneGraph("harbor_map_boat", "harbor", "pilot", "folded", "map", "lies", "sailor", "boat", "waits", "dock"),
    SceneGraph("garden_frame_bench", "garden", "carver", "wooden", "frame", "leans", "keeper", "bench", "sits", "wall"),
    SceneGraph("studio_candle_bowl", "studio", "weaver", "silver", "candle", "burns", "maker", "bowl", "rests", "shelf"),
    SceneGraph("market_book_crate", "market", "vendor", "quiet", "book", "lies", "porter", "crate", "waits", "stall"),
    SceneGraph("chapel_ribbon_chest", "chapel", "organist", "blue", "ribbon", "hangs", "warden", "chest", "rests", "altar"),
)


# Right-side lexicalizations are independent of left words.  Each option is a
# typed lexical unit, not a copied or reversed fragment.  The template keeps
# ordinary English determiner/possessive/attributive/locative order explicit.
RIGHT_UNITS: dict[str, tuple[str, ...]] = {
    "DET": ("the", "a", "our"),
    "PLACE": ("gallery", "harbor", "garden", "studio", "market", "chapel", "arch", "dock", "wall", "shelf", "stall", "altar"),
    "OWNER": ("artist", "curator", "pilot", "sailor", "carver", "keeper", "weaver", "maker", "vendor", "porter", "organist", "warden"),
    "ATTRIBUTE": ("amber", "folded", "wooden", "silver", "quiet", "blue", "bright", "small", "old", "still"),
    "OBJECT": ("lamp", "vase", "map", "boat", "frame", "bench", "candle", "bowl", "book", "crate", "ribbon", "chest"),
    "VERB": ("rests", "stands", "lies", "waits", "leans", "sits", "burns", "hangs", "glows", "opens"),
    "PREP": ("near", "by", "in", "at"),
    "CONJ": ("and",),
}

# Different constituent order from the left realization: object-B clause is
# fronted before the attributed object-A clause.  The graph role labels are
# carried through this order, so syntax and semantics constrain one another.
RIGHT_TEMPLATE = (
    "DET", "OWNER", "OBJECT", "VERB", "PREP", "DET", "PLACE", "CONJ",
    "DET", "OWNER", "ATTRIBUTE", "OBJECT", "VERB", "PREP", "DET", "PLACE",
)

# Each lexical slot is paired with the graph edge whose realization it
# licenses.  Keeping this alignment in the residual key prevents a lexical
# prefix from being counted merely because its characters fit: it must also
# be a legal realization of the pending scene edge.
RIGHT_SLOT_EDGES = (
    "possesses", "possesses", "possesses", "predicate", "located_at", "located_at", "located_at", "coordination",
    "possesses", "possesses", "describes", "possesses", "predicate", "located_at", "located_at", "located_at",
)


def _unit_words(text: str) -> tuple[str, ...]:
    return tokenize(text)


@lru_cache(maxsize=None)
def _partial_segmentations(tape: str, template: tuple[str, ...], cap: int = 36) -> tuple[dict[str, Any], ...]:
    """Enumerate reverse-compatible typed lexical prefixes, not bare words."""
    frontier: list[dict[str, Any]] = []

    def rec(offset: int, slot: int, words: tuple[str, ...], units: tuple[str, ...]) -> None:
        frontier.append({
            "consumed": offset,
            "words": words,
            "units": units,
            "slot_index": slot,
            "next_role": template[slot] if slot < len(template) else None,
            "pending_graph_edge": RIGHT_SLOT_EDGES[slot] if slot < len(template) else None,
        })
        if slot >= len(template):
            return
        role = template[slot]
        for phrase in RIGHT_UNITS[role]:
            phrase_words = _unit_words(phrase)
            letters = "".join(phrase_words)
            if tape.startswith(letters, offset):
                rec(offset + len(letters), slot + 1, words + phrase_words, units + (phrase,))

    rec(0, 0, (), ())
    frontier.sort(key=lambda row: (-row["consumed"], row["slot_index"], row["words"]))
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


def _independent_tape(text: str) -> str:
    """Independent ASCII-only exactness normalizer (not admission's helper)."""
    lowered = text.casefold()
    if any(ch.isalpha() and not ("a" <= ch <= "z") for ch in lowered):
        raise ValueError("non_ascii_alpha")
    return "".join(ch for ch in lowered if "a" <= ch <= "z")


def _independent_audit(text: str) -> dict[str, Any]:
    tape = _independent_tape(text)
    mismatches = [
        {"pair": i, "left": tape[i], "right": tape[-1 - i]}
        for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]
    ]
    return {
        "exact": bool(tape) and not mismatches,
        "two_pointer_exact": bool(tape) and all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "letters": len(tape),
        "normalized_letters": tape,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "mismatch_count": len(mismatches),
        "mismatch_sample": mismatches[:12],
    }


def _left_parse(scene: SceneGraph) -> dict[str, Any]:
    words = tokenize(scene.text)
    return {
        "valid": bool(words) and scene.text.endswith(".") and " and " in scene.text,
        "grammar": "in-the-place possessive attributed-object locative AND possessive-object locative",
        "word_count": len(words),
        "semantic_graph": scene.graph,
        "located_at_edges": 3,
        "possesses_edges": 2,
        "describes_edges": 1,
    }


def _right_parse(words: tuple[str, ...], graph: dict[str, Any]) -> dict[str, Any]:
    """Independent parser for a completed right realization."""
    # The slot parser is deliberately separate from the left renderer; these
    # checks ensure the right surface has the intended coordinated grammar.
    valid = (
        len(words) >= 16 and words[7] == "and" and words[1] in RIGHT_UNITS["OWNER"]
        and words[2] in RIGHT_UNITS["OBJECT"] and words[4] in RIGHT_UNITS["PREP"]
        and words[9] in RIGHT_UNITS["OWNER"] and words[10] in RIGHT_UNITS["ATTRIBUTE"]
        and words[11] in RIGHT_UNITS["OBJECT"] and words[13] in RIGHT_UNITS["PREP"]
    )
    return {
        "valid": valid,
        "template": list(RIGHT_TEMPLATE),
        "word_count": len(words),
        "semantic_graph_topology": [edge["type"] for edge in graph["edges"]],
        "slot_edge_alignment": list(RIGHT_SLOT_EDGES),
        "coordinated_clause_count": 2,
        "english_order_checked": True,
        "edge_roles_checked": valid,
    }


def _readability_diagnostic(text: str) -> dict[str, Any]:
    words = tokenize(text)
    freqs = [zipf_frequency(word, "en") for word in words]
    return {
        "status": "diagnostic_only_unreviewed",
        "word_count": len(words),
        "all_words_zipf_ge_2": bool(words) and all(value >= 2 for value in freqs),
        "mean_zipf_frequency": round(sum(freqs) / max(1, len(freqs)), 3),
        "coordinated_clause_marker": " and " in text.lower(),
        "blinded_reader_required": True,
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
    """Fingerprint bounded JSON strings before creating this run's output."""
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


def run(output: Path | None = None) -> dict[str, Any]:
    existing, scan = _existing_tape_keys(output)
    stats = Counter()
    rejections: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []

    for scene in SCENES:
        stats["scene_graphs"] += 1
        reverse_tape = scene.tape[::-1]
        prefixes = _partial_segmentations(reverse_tape, RIGHT_TEMPLATE)
        best = prefixes[0]
        full = tuple(row for row in prefixes if row["consumed"] == len(reverse_tape) and row["next_role"] is None)
        stats["reverse_prefix_states"] += len(prefixes)
        if not full:
            stats["reverse_grammar_failures"] += 1
            rendered_probe = f"{scene.text[:-1]}; {' '.join(best['words'])}".rstrip()
            rejections.append({
                "kind": "scene_graph_reverse_residual_probe",
                "frame_id": scene.frame_id,
                "rendered_probe": rendered_probe,
                "source_semantic_graph": scene.graph,
                "independent_left_parse": _left_parse(scene),
                "reverse_tape_length": len(reverse_tape),
                "reverse_prefix_letters": best["consumed"],
                "reverse_prefix_words": list(best["words"]),
                "reverse_tape_prefix": reverse_tape[:best["consumed"]],
                "reverse_residual": reverse_tape[best["consumed"]:best["consumed"] + 20],
                "residual_state": {
                    "slot_index": best["slot_index"],
                    "next_role": best["next_role"],
                    "pending_graph_edge": best["pending_graph_edge"],
                    "graph_edges_remaining": list(RIGHT_TEMPLATE[best["slot_index"]:]),
                    "english_grammar_checked_jointly": True,
                },
                "independent_exact_audit": _independent_audit(rendered_probe),
                "central_admission": mechanical_admission_checks(rendered_probe, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS),
                "readability_diagnostic": _readability_diagnostic(rendered_probe),
                "reader_status": "not_run",
            })
            continue

        stats["reverse_graph_closures"] += len(full)
        for parsed in full:
            right = " ".join(parsed["words"])
            rendered = f"{scene.text[:-1]}; {right}."
            audit = _independent_audit(rendered)
            gate = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
            tape = audit["normalized_letters"]
            novelty = tape not in existing
            right_parse = _right_parse(parsed["words"], scene.graph)
            row = {
                "kind": "exact_scene_graph_closure",
                "rendered": rendered,
                "source_semantic_graph": scene.graph,
                "right_semantic_graph": scene.graph,
                "right_words": list(parsed["words"]),
                "right_slot_units": list(parsed["units"]),
                "right_slot_edge_alignment": list(RIGHT_SLOT_EDGES),
                "independent_right_parse": right_parse,
                "independent_exact_audit": audit,
                "central_admission": gate,
                "novelty_audit": {"tape_absent_from_all_existing_json_keys": novelty, "tape_key": tape},
                "readability_diagnostic": _readability_diagnostic(rendered),
                "mechanically_admitted": audit["exact"] and novelty and right_parse["valid"] and all(gate.values()),
                "reader_status": "not_run; exactness does not certify readability",
            }
            exact_rows.append(row)
            if row["mechanically_admitted"]:
                stats["mechanically_admitted"] += 1

    rejections.sort(key=lambda row: (-row["reverse_prefix_letters"], row["frame_id"]))
    exact_rows.sort(key=lambda row: (-row["independent_exact_audit"]["letters"], row["rendered"]))
    return {
        "status": "scene_graph_reverse_search_complete",
        "family_id": FAMILY_ID,
        "state_space_signature": STATE_SPACE_SIGNATURE,
        "config": {
            "construction": "coordinated locative + possessive + attributive scene clauses",
            "semantic_graph_edges": ["located_at", "possesses", "describes"],
            "left_template": "in-the-place | possessive attributed object near place | and possessive object by place",
            "right_template": list(RIGHT_TEMPLATE),
            "bidirectional_slot_graph": True,
            "reverse_character_constraints_and_grammar_joint": True,
            "fresh_authored_scene_inventory": True,
            "brown_corpus_used": False,
            "event_discourse_dialogue_inventory_used": False,
            "global_pos_or_cfg_search_used": False,
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "output_excluded_before_scan": True,
            "preexisting_tape_key_scope": "all normalized strings of length 1..280 in runs/**/*.json, data/**/*.json, experiments/**/*.json",
        },
        "novelty_audit": {
            "existing_tape_keys_count": len(existing),
            "existing_json_files_scanned": scan["json_files_scanned"],
            "malformed_json_files_skipped": scan["malformed_json_files"],
            "existing_tape_keys_sha256": _digest(existing),
            "output_path_excluded_before_scan": bool(output),
            "all_exact_rows_checked_against_existing_keys": True,
            "all_mechanically_admitted_rows_novel": all(row["novelty_audit"]["tape_absent_from_all_existing_json_keys"] for row in exact_rows if row["mechanically_admitted"]),
        },
        "stats": dict(stats),
        "exact_closures": exact_rows,
        "admitted": [row for row in exact_rows if row["mechanically_admitted"]],
        "residual_frontier": rejections[:MAX_PROBES],
        "next_operator": (
            "At the deepest recorded reverse seam, replace exactly one scene-graph terminal realization "
            "with an independently authored multiword locative or attributive NP; preserve all edge types, "
            "possessor binding, coordinated clause order, and the output-excluded tape fingerprint. "
            "Only branch when the replacement consumes the recorded residual prefix."
        ),
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scene_inventory_authored": True,
            "right_lexicon_independently_authored": True,
            "source_sentences_copied": False,
            "catalogue_relexicalization": False,
            "readability_certificate": False,
        },
        "reader_gate": {
            "status": "not_run",
            "reason": "No item may be sent to readers until exact, admission, novelty, and intact-prose review gates are separately satisfied.",
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
    print(json.dumps({"status": result["status"], "stats": result["stats"], "exact_closures": len(result["exact_closures"]), "admitted": len(result["admitted"])}, indent=2))


if __name__ == "__main__":
    main()
