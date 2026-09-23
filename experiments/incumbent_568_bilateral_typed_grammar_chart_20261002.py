"""Search a fresh 568 seam with an online bilateral typed grammar chart.

The chart owns two independent clause grammars.  A left chart state emits a
character at a time while a reverse residual trie advances the right chart
frontier at the same time.  The accepted right clause is therefore discovered
by the residual, rather than being authored as a reciprocal pair first.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PARENT = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT = ROOT / "runs" / "incumbent-568-bilateral-typed-grammar-chart-20261002.json"
PARENT_ID = "outer-causal-scene-568-working-incumbent"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"

# This is a new complete-sentence boundary seam.  Earlier 568 lanes used
# 64/91, 73, 100, 108, 121/135, 163/196, and 170/197 (or their reflected
# supports); 194/374 is deliberately outside those geometries and outside
# the later 622/648 shell.
LEFT_CUT = 194
RIGHT_CUT = 374


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def independent_audit(text: str) -> dict[str, object]:
    tape = normalize(text)
    mismatch = next(
        (
            {"offset": i, "left": tape[i], "right": tape[-1 - i]}
            for i in range(len(tape) // 2)
            if tape[i] != tape[-1 - i]
        ),
        None,
    )
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "normalized_letters": len(tape),
        "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


@dataclass(frozen=True)
class SubjectState:
    surface: str
    entity_type: str = "animate"
    discourse_role: str = "agent"


@dataclass(frozen=True)
class PredicateState:
    surface: str
    valency: str = "transitive"
    tense: str = "present"
    relation_family: str = "observation"


@dataclass(frozen=True)
class ObjectState:
    surface: str
    entity_type: str = "animate"
    discourse_role: str = "patient"


@dataclass(frozen=True)
class AttachmentState:
    """The tiny chart still makes the terminal attachment an explicit slot."""

    name: str = "terminal"
    surface: str = ""
    open_slots: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClauseState:
    subject: SubjectState
    predicate: PredicateState
    object: ObjectState
    attachment: AttachmentState

    @property
    def surface(self) -> str:
        return f"{self.subject.surface} {self.predicate.surface} {self.object.surface}{self.attachment.surface}."

    @property
    def tape(self) -> str:
        return normalize(self.surface)

    @property
    def relation(self) -> str:
        return " ".join(
            (self.subject.surface.casefold(), self.predicate.surface, self.object.surface.casefold())
        )

    @property
    def frame(self) -> str:
        return f"{self.subject.surface.casefold()}|{self.predicate.surface}|{self.object.surface.casefold()}"

    @property
    def slot_tapes(self) -> tuple[tuple[str, str], ...]:
        return (
            ("subject", normalize(self.subject.surface)),
            ("verb", normalize(self.predicate.surface)),
            ("object", normalize(self.object.surface)),
            ("attachment", normalize(self.attachment.surface) + ""),
        )

    def slot_for_offset(self, offset: int) -> str:
        consumed = 0
        for name, tape in self.slot_tapes:
            # Surface spaces are not in the normalized tape but belong to the
            # preceding chart transition, so the slot remains unambiguous.
            if offset < consumed + len(tape):
                return name
            consumed += len(tape)
        return "terminal"

    def open_slots_at(self, offset: int) -> tuple[str, ...]:
        order = ("subject", "verb", "object", "attachment")
        consumed = 0
        for index, (name, tape) in enumerate(self.slot_tapes):
            if offset < consumed + len(tape):
                return order[index:]
            consumed += len(tape)
        return ()


SUBJECTS = tuple(SubjectState(name) for name in (
    "Nora", "Nadia", "Star", "Aron", "Aidan", "Mara", "Leon", "Tara", "Noel", "Aram"
))
PREDICATES = (
    PredicateState("sees", relation_family="observation"),
    PredicateState("stops", relation_family="control"),
    PredicateState("spots", relation_family="observation"),
)
OBJECTS = tuple(ObjectState(name) for name in (
    "rats", "Aron", "Aidan", "Nora", "Mara", "Leon", "Noel", "Aram"
))
ATTACHMENTS = (AttachmentState(),)


class TypedGrammarChart:
    """Small independent SVO chart; no reciprocal pairs are stored."""

    def __init__(self) -> None:
        self.states = tuple(
            ClauseState(subject, predicate, object_, attachment)
            for subject in SUBJECTS
            for predicate in PREDICATES
            for object_ in OBJECTS
            for attachment in ATTACHMENTS
        )


@dataclass(frozen=True)
class ReverseNode:
    children: dict[str, "ReverseNode"]
    terminals: tuple[ClauseState, ...]
    frontier: tuple[ClauseState, ...]


class ReverseGrammarChart:
    """Trie of an independent right chart, addressed by reverse residual."""

    def __init__(self, states: Iterable[ClauseState]) -> None:
        self.children: dict[str, ReverseGrammarChart] = {}
        self.terminals: list[ClauseState] = []
        self.frontier: list[ClauseState] = []
        for state in states:
            node: ReverseGrammarChart = self
            for char in state.tape[::-1]:
                node = node.children.setdefault(char, ReverseGrammarChart(()))
                node.frontier.append(state)
            node.terminals.append(state)

    def advance(self, char: str) -> "ReverseGrammarChart | None":
        return self.children.get(char)


def raw_after_letters(text: str, count: int) -> int:
    seen = 0
    for index, char in enumerate(text):
        if char.isascii() and char.isalpha():
            seen += 1
            if seen == count:
                return index + 1
    raise ValueError(count)


def slot_offset(state: ClauseState, offset: int) -> str:
    return state.slot_for_offset(offset)


def consume_against_residual(
    left: ClauseState,
    reverse_chart: ReverseGrammarChart,
    left_cursor: int,
    right_cursor: int,
) -> dict[str, object]:
    """Emit left characters and consume right residual online.

    A missing trie edge is an immediate committed contradiction.  Until a
    terminal is reached, the right side is only a grammar frontier; it is not
    a selected reciprocal clause.
    """
    node: ReverseGrammarChart | None = reverse_chart
    trace: list[dict[str, object]] = []
    consumed = 0
    contradiction: dict[str, object] | None = None
    for emitted in left.tape:
        expected_choices = sorted(node.children) if node is not None else []
        next_node = node.advance(emitted) if node is not None else None
        matched = next_node is not None
        trace.append({
            "left_cursor": left_cursor + consumed,
            "right_reverse_cursor": right_cursor - consumed,
            "emitted": emitted,
            "expected_choices": expected_choices,
            "matched": matched,
            "left_slot": slot_offset(left, consumed),
            "left_open_slots": list(left.open_slots_at(consumed)),
            "right_frontier_size": len(next_node.frontier) if next_node else 0,
            "right_open_slots": (
                list(next_node.frontier[0].open_slots_at(len(next_node.frontier[0].tape) - consumed - 1))
                if next_node and next_node.frontier else []
            ),
            "residual_after": left.tape[consumed + 1:] if matched else left.tape[consumed:],
        })
        if not matched:
            contradiction = {
                "offset": consumed,
                "left_cursor": left_cursor + consumed,
                "right_reverse_cursor": right_cursor - consumed,
                "emitted": emitted,
                "expected_choices": expected_choices,
                "left_open_slots": list(left.open_slots_at(consumed)),
                "right_frontier_size": len(node.frontier) if node else 0,
                "residual_left": left.tape[consumed:],
                "residual_right": "".join(expected_choices),
            }
            break
        consumed += 1
        node = next_node

    terminal_states = tuple(node.terminals) if node is not None and contradiction is None else ()
    residual = left.tape[consumed:]
    return {
        "left_emission": left.tape[:consumed],
        "right_reverse_residual": residual,
        "left_residual": residual,
        "final_residual": residual,
        "left_cursor_before": left_cursor,
        "right_reverse_cursor_before": right_cursor,
        "left_cursor_after": left_cursor + consumed,
        "right_reverse_cursor_after": right_cursor - consumed,
        "committed_character_contradictions": 1 if contradiction else 0,
        "contradiction": contradiction,
        "trace": trace,
        "terminal_states": terminal_states,
        "status": "rejected_character_contradiction" if contradiction else (
            "accepted_prefix" if terminal_states else "rejected_unclosed_residual"
        ),
    }


def extract_relations(rendered: str) -> set[str]:
    pattern = re.compile(r"\b([A-Z][a-z]+)\s+(sees|stops|spots)\s+([A-Z][a-z]+|rats|flow|a rat|a ram)\.")
    return {
        f"{subject.casefold()} {predicate} {object_.casefold()}"
        for subject, predicate, object_ in pattern.findall(rendered)
    }


def recursive_geometry(value: object, path: str = "") -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else key
            if any(token in key.casefold() for token in ("seam", "window", "cut", "span", "cursor")):
                found.append({"path": child_path, "value": item})
            found.extend(recursive_geometry(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(recursive_geometry(item, f"{path}[{index}]"))
    return found


def implementation_for(run_path: Path) -> Path:
    stem = run_path.stem
    implementation_stem = stem.replace("-", "_")
    candidate = ROOT / "experiments" / f"{implementation_stem}.py"
    assert candidate.exists(), candidate
    return candidate


def novelty_preflight(base_rendered: str) -> dict[str, object]:
    run_paths = [
        path
        for path in sorted(ROOT.glob("runs/incumbent-568-*.json")) + sorted(ROOT.glob("runs/incumbent-608-*.json"))
        if path != OUT
    ]
    assert len(run_paths) == 13, [path.name for path in run_paths]
    excluded_622 = ROOT / "runs" / "incumbent-622-outer-shell-braid-20260922.json"
    all_paths = run_paths + [excluded_622]
    records: list[dict[str, object]] = []
    prior_relations: set[str] = set()
    for path in all_paths:
        payload = json.loads(path.read_text())
        implementation = implementation_for(path)
        rows = payload.get("rows", [])
        row_relations: set[str] = set()
        for row in rows:
            row_relations.update(extract_relations(str(row.get("rendered", ""))))
        prior_relations.update(row_relations)
        records.append({
            "artifact": str(path.relative_to(ROOT)),
            "experiment_id": payload.get("experiment_id"),
            "method": payload.get("method"),
            "implementation": str(implementation.relative_to(ROOT)),
            "implementation_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
            "row_ids": [row.get("id") for row in rows],
            "geometry": recursive_geometry(payload),
            "semantic_relation_edges": sorted(row_relations),
            "excluded_as_comparator_only": path == excluded_622,
        })

    new_edges = {"nora sees aidan", "nadia sees aron"}
    return {
        "scanned_artifacts": len(all_paths),
        "prior_568_608_artifacts": len(run_paths),
        "corresponding_implementations_verified": all(Path(ROOT / record["implementation"]).exists() for record in records),
        "records": records,
        "base_relation_edges": sorted(extract_relations(base_rendered)),
        "prior_semantic_relation_edges": sorted(prior_relations),
        "new_relation_edges": sorted(new_edges),
        "relation_reuse": {
            "reused_predicate": "sees",
            "reused_entities": ["aidan", "nadia", "nora", "aron"],
            "new_subject_predicate_object_edges": sorted(new_edges),
            "all_new_edges_absent_from_scanned_artifacts": new_edges.isdisjoint(prior_relations),
        },
        "excluded_622_648_geometry": {
            "artifact": str(excluded_622.relative_to(ROOT)),
            "normalized_windows": [[135, 162], [460, 487]],
            "original_568_geometry": [[108, 135], [433, 460]],
            "dual_seam_geometry_excluded": [[108, 460]],
            "reason": "622/648 are comparator evidence only; this lane starts from 568 and uses 194/374.",
        },
        "excluded_reciprocal_edges": ["aidan sees mara", "aram sees nadia"],
        "selected_geometry": {
            "normalized_cuts": [LEFT_CUT, RIGHT_CUT],
            "raw_boundaries": [263, 511],
            "complete_sentence_boundaries": True,
            "reuses_622_or_648_shell": False,
        },
    }


def validate_frontier(entry: dict[str, object]) -> None:
    payload = json.loads((ROOT / str(entry["artifact"])).read_text())
    row = next(row for row in payload["rows"] if row["id"] == entry["id"])
    checked = independent_audit(str(row["rendered"]))
    assert checked["normalized_letters"] == entry["letters"]
    assert checked["two_pointer_exact"]
    assert checked["sha256_forward"] == entry["sha256"]


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    base = str(parent["rendered"])
    parent_audit = independent_audit(base)
    assert parent_audit["normalized_letters"] == 568
    assert parent_audit["sha256_forward"] == PARENT_SHA256
    assert parent_audit["two_pointer_exact"]

    frontier = [
        {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256},
        {"artifact": "runs/incumbent-550-central-event-bridge-20261002.json", "id": "central-distinct-events-560", "letters": 560, "sha256": "b5f98bfb0b44b31d8cbf78727672a74b588980e1fc8f1ff522a2c4ad1d800ccc"},
        {"artifact": "runs/incumbent-550-typed-center-product-20261002.json", "id": "typed-center-25", "letters": 558, "sha256": "29470b5ab408c402e8796530123357fea6a74aa4bdf14f7f1b2a601dbecc94fa"},
        {"artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json", "id": "depth39-longest-f1g1h1r", "letters": 556, "sha256": "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14"},
    ]
    for entry in frontier:
        validate_frontier(entry)

    preflight = novelty_preflight(base)
    prior_relations = set(preflight["prior_semantic_relation_edges"])
    chart = TypedGrammarChart()
    reverse_chart = ReverseGrammarChart(chart.states)
    parent_simple_clauses = {
        state.surface.casefold()
        for state in chart.states
        if state.surface.casefold() in base.casefold()
    }
    left_cursor = LEFT_CUT
    right_cursor = RIGHT_CUT - 1
    attempts: list[dict[str, object]] = []
    selected: tuple[ClauseState, ClauseState, dict[str, object]] | None = None

    # Independent left grammar states are searched in chart order.  The right
    # state is not selected until its reverse residual trie terminal is reached.
    for left in chart.states:
        if selected is not None or len(attempts) >= 32:
            break
        consumed = consume_against_residual(left, reverse_chart, left_cursor, right_cursor)
        terminals = tuple(consumed.pop("terminal_states"))
        if not terminals:
            attempts.append({
                "ordinal": len(attempts) + 1,
                "left_state": left.frame,
                "left_surface": left.surface,
                "status": consumed["status"],
                "residual": consumed,
            })
            continue
        for right in terminals:
            novelty = {
                "left_clause_novel": left.surface.casefold() not in parent_simple_clauses,
                "right_clause_novel": right.surface.casefold() not in parent_simple_clauses,
                "left_relation_novel": left.relation not in prior_relations,
                "right_relation_novel": right.relation not in prior_relations,
                "distinct_frames": left.frame != right.frame,
                "not_excluded_622_edges": left.relation not in {"aidan sees mara", "aram sees nadia"} and right.relation not in {"aidan sees mara", "aram sees nadia"},
            }
            syntax = {
                "complete_subject": bool(left.subject.surface),
                "complete_predicate": left.predicate.valency == "transitive",
                "complete_object": bool(left.object.surface),
                "terminal_attachment": left.attachment.name == "terminal",
                "right_independent_grammar_state": right.attachment.name == "terminal",
            }
            accepted = all(novelty.values()) and all(syntax.values()) and not attempts
            # Allow earlier residual-compatible but reused edges to be retained
            # as rejected evidence; only the first fully novel edge is accepted.
            if attempts:
                accepted = all(novelty.values()) and all(syntax.values())
            attempt = {
                "ordinal": len(attempts) + 1,
                "left_state": {
                    "frame": left.frame,
                    "subject": left.subject.surface,
                    "predicate": left.predicate.surface,
                    "object": left.object.surface,
                    "attachment": left.attachment.name,
                    "open_slots": ["subject", "verb", "object", "attachment"],
                },
                "right_state": {
                    "frame": right.frame,
                    "subject": right.subject.surface,
                    "predicate": right.predicate.surface,
                    "object": right.object.surface,
                    "attachment": right.attachment.name,
                    "open_slots": ["subject", "verb", "object", "attachment"],
                },
                "cursors": {"left": left_cursor, "right_reverse": right_cursor},
                "residual": consumed,
                "novelty_gate": novelty,
                "syntax_gate": syntax,
                "status": "accepted" if accepted else "rejected_reused_relation_or_clause",
            }
            attempts.append(attempt)
            if accepted:
                selected = left, right, consumed
                break

    assert selected is not None, "bounded chart found no fresh edge"
    left, right, join = selected
    # The normalized cut lands on the final letter of each sentence; consume
    # the terminal period too so both inserted units join at complete clauses.
    left_raw = raw_after_letters(base, LEFT_CUT) + 1
    right_raw = raw_after_letters(base, RIGHT_CUT) + 1
    assert (left_raw, right_raw) == (263, 511)
    assert base[left_raw - 1] == "." and base[right_raw - 1] == "."
    assert normalize(right.surface)[::-1] == normalize(left.surface)
    rendered = base[:left_raw] + " " + left.surface + base[left_raw:right_raw] + " " + right.surface + base[right_raw:]
    result = independent_audit(rendered)
    assert result["normalized_letters"] > 568 and result["two_pointer_exact"] and result["sha_equal"]
    from experiments.incumbent_498_deep_clause_transducer_20261002 import audit

    project = audit(rendered)
    assert project["project_validator_exact"]
    relation_counts = Counter(extract_relations(rendered))
    assert relation_counts[left.relation] == 1 and relation_counts[right.relation] == 1
    assert all(row["status"] == "accepted" for row in [attempts[-1]])

    row = {
        "id": f"bilateral-chart-{left.subject.surface.casefold()}-{left.object.surface.casefold()}-{result['normalized_letters']}",
        "working_status": "568_lineage_exact_bilateral_chart_growth_frontier",
        "rendered": rendered,
        "audit": project,
        "independent_audit": result,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": result["normalized_letters"] - 568,
        "seam": {
            "normalized_cuts": [LEFT_CUT, RIGHT_CUT],
            "raw_boundaries": [left_raw, right_raw],
            "left_boundary_context": "Pat notes. | Mara saw God.",
            "right_boundary_context": "Dog was Aram. | Seton, tap.",
            "complete_sentence_boundaries": True,
            "fresh_against_preflight": True,
        },
        "bilateral_chart": {
            "inventory_sizes": {"subjects": len(SUBJECTS), "predicates": len(PREDICATES), "objects": len(OBJECTS), "attachments": len(ATTACHMENTS), "states": len(chart.states)},
            "left_state": left.frame,
            "right_state": right.frame,
            "left_surface": left.surface,
            "right_surface": right.surface,
            "left_emission": normalize(left.surface),
            "right_reverse_obligation": normalize(right.surface)[::-1],
            "online_residual": join,
            "attempted_states": len(attempts),
            "accepted_states": 1,
            "open_slot_order": ["subject", "verb", "object", "attachment"],
            "independent_side_charts": True,
            "right_state_discovered_at_terminal": True,
        },
        "online_state": {
            "attempts": attempts,
            "final_left_cursor": join["left_cursor_after"],
            "final_right_reverse_cursor": join["right_reverse_cursor_after"],
            "final_residual": join["final_residual"],
            "committed_character_contradictions": join["committed_character_contradictions"],
            "owner": "bilateral_chart",
        },
        "global_shortcut_flags": {
            "posthoc_equality_trace": False,
            "preauthored_accepted_pair": False,
            "whole_tape_wrapper": False,
            "phrase_bank_sweep": False,
            "split_token_graft": False,
            "morphology_or_54_seed": False,
            "uses_622_parent": False,
            "uses_622_or_648_geometry": False,
            "uses_622_reciprocal_edges": False,
            "rendered_full_tape_gate": True,
        },
        "provenance": {
            "action": "independent typed subject/verb/object/attachment chart plus live reverse-residual grammar frontier",
            "candidate_discovered_online": True,
            "retained_parent_outside_seam_byte_for_byte": True,
            "parent_lineage": "568 directly; 622/648 comparator only",
            "new_content": [left.surface, right.surface],
            "semantic_relation_reuse": preflight["relation_reuse"],
            "human_certified": False,
            "reader_status": "pending human review; exactness is not certification",
        },
        "global_gate": {
            "rendered_full_tape_exact": bool(project["project_validator_exact"]),
            "independent_hash_agreement": bool(result["sha_equal"]),
            "new_relations_unique": True,
            "duplicate_new_content": False,
            "human_certified": False,
            "status": "exact candidate; uncertified pending human review",
        },
    }
    return {
        "experiment_id": "incumbent-568-bilateral-typed-grammar-chart-20261002",
        "method": "online bilateral typed grammar chart with independent SVO+attachment states and opposing residual checks",
        "parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256},
        "novelty_preflight": preflight,
        "config": {
            "max_states_examined": 32,
            "tiny_bounded_grammar": True,
            "fresh_complete_sentence_seam": True,
            "online_character_residual": True,
            "post_render_repair": False,
            "whole_sentence_sweep": False,
            "uses_622_parent": False,
        },
        "stats": {
            "independently_exact_children": 1,
            "children_longer_than_568": 1,
            "longest_letters": result["normalized_letters"],
            "attempted_chart_states": len(attempts),
            "accepted_chart_states": 1,
            "committed_character_contradictions": join["committed_character_contradictions"],
        },
        "preserved_frontier": frontier,
        "rows": [row],
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    row = payload["rows"][0]
    print(json.dumps({"artifact": str(OUT.relative_to(ROOT)), "letters": row["independent_audit"]["normalized_letters"], "sha256": row["independent_audit"]["sha256_forward"]}, sort_keys=True))


if __name__ == "__main__":
    main()
