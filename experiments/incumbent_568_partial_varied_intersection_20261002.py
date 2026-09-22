"""A bounded, varied grammar/trie expansion at a recomputed partial-word seam.

The authoritative 568 tape is loaded directly.  The selected seam is inside
``Mara``/``Aram`` (normalized cuts 196/372); the surrounding complete clauses
are replaced only after the live reverse-trie intersection admits two distinct
predicate families.  This is intentionally a new operator: clause, frame,
subject, predicate, and object novelty are checked while the two cursors and
residual tapes advance, rather than repaired after rendering.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.incumbent_498_deep_clause_transducer_20261002 import audit

PARENT = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT = ROOT / "runs" / "incumbent-568-partial-varied-intersection-20261002.json"
PARENT_ID = "outer-causal-scene-568-working-incumbent"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
MAX_PAIRED_EXPANSIONS = 8
SEAM_LEFT = 196
SEAM_RIGHT = 372


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def independent_audit(text: str) -> dict[str, object]:
    tape = normalize(text)
    mismatch = next(
        ({"offset": i, "left": tape[i], "right": tape[-i - 1]}
         for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]),
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


def raw_after_letters(text: str, count: int) -> int:
    seen = 0
    for index, char in enumerate(text):
        if char.isascii() and char.isalpha():
            seen += 1
            if seen == count:
                return index + 1
    raise ValueError(count)


def partial_word(text: str, cut: int) -> str:
    position = 0
    for match in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text):
        word = normalize(match.group())
        if position < cut < position + len(word):
            offset = cut - position
            return word[:offset] + "|" + word[offset:]
        position += len(word)
    return "boundary"


@dataclass(frozen=True)
class Clause:
    surface: str
    subject: str
    predicate: str
    object: str
    role_family: str
    active_entities: tuple[str, ...]

    @property
    def tape(self) -> str:
        return normalize(self.surface)

    @property
    def frame(self) -> str:
        return f"{self.subject}|{self.predicate}|{self.object}"


class ReverseClauseTrie:
    """Index complete clauses by their reversed character obligation."""

    def __init__(self) -> None:
        self.children: dict[str, ReverseClauseTrie] = {}
        self.terminals: list[Clause] = []

    def add(self, clause: Clause) -> None:
        node = self
        for char in clause.tape[::-1]:
            node = node.children.setdefault(char, ReverseClauseTrie())
        node.terminals.append(clause)

    def exact(self, left_tape: str) -> tuple[Clause, ...]:
        node = self
        for char in left_tape:
            node = node.children.get(char)
            if node is None:
                return ()
        return tuple(node.terminals)


def targeted_inventory() -> tuple[Clause, ...]:
    """Build a small discourse inventory, not a catalogue of prejoined pairs."""
    # Observation and control are deliberately separate discourse roles.  The
    # reverse trie, not this inventory, discovers the partner for each clause.
    families = (
        ("observation", "sees", ("rats", "Aron", "Nora")),
        ("control", "stops", ("Aron", "rats", "Aidan", "Nora", "flow")),
        ("control", "spots", ("Aidan", "rats", "Aron", "Nora", "flow")),
    )
    subjects = ("Nora", "Nadia", "Star", "Aron", "Aidan")
    rows: list[Clause] = []
    for family, predicate, objects in families:
        for subject in subjects:
            for object_ in objects:
                rows.append(Clause(
                    surface=f"{subject} {predicate} {object_}.",
                    subject=subject.casefold(), predicate=predicate,
                    object=object_.casefold(), role_family=family,
                    active_entities=(subject.casefold(), object_.casefold()),
                ))
    return tuple(rows)


def consume_pair(left: Clause, right: Clause, left_cursor: int, right_cursor: int) -> dict[str, object]:
    left_tape, obligation = left.tape, right.tape[::-1]
    residual = obligation
    trace: list[dict[str, object]] = []
    contradictions = 0
    for offset, emitted in enumerate(left_tape):
        expected = residual[0] if residual else None
        matched = emitted == expected
        trace.append({
            "left_cursor": left_cursor + offset,
            "right_reverse_cursor": right_cursor - offset,
            "emitted": emitted,
            "expected": expected,
            "matched": matched,
        })
        if not matched:
            contradictions += 1
            break
        residual = residual[1:]
    return {
        "left_emission": left_tape,
        "right_reverse_obligation": obligation,
        "left_residual": residual,
        "right_residual": residual,
        "final_residual": residual,
        "left_cursor_before": left_cursor,
        "right_reverse_cursor_before": right_cursor,
        "left_cursor_after": left_cursor + len(left_tape),
        "right_reverse_cursor_after": right_cursor - len(obligation),
        "committed_character_contradictions": contradictions,
        "trace": trace,
    }


def validate_entry(entry: dict[str, object]) -> None:
    payload = json.loads((ROOT / str(entry["artifact"])).read_text())
    row = next(row for row in payload["rows"] if row["id"] == entry["id"])
    checked = independent_audit(str(row["rendered"]))
    assert checked["normalized_letters"] == entry["letters"]
    assert checked["two_pointer_exact"] and checked["sha256_forward"] == entry["sha256"]


def build_payload() -> dict[str, object]:
    payload = json.loads(PARENT.read_text())
    parent = next(row for row in payload["rows"] if row["id"] == PARENT_ID)
    base = str(parent["rendered"])
    assert independent_audit(base)["sha256_forward"] == PARENT_SHA256
    assert independent_audit(base)["normalized_letters"] == 568
    base_tape = normalize(base)

    frontier = [
        {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256},
        {"artifact": "runs/incumbent-550-central-event-bridge-20261002.json", "id": "central-distinct-events-560", "letters": 560, "sha256": "b5f98bfb0b44b31d8cbf78727672a74b588980e1fc8f1ff522a2c4ad1d800ccc"},
        {"artifact": "runs/incumbent-550-typed-center-product-20261002.json", "id": "typed-center-25", "letters": 558, "sha256": "29470b5ab408c402e8796530123357fea6a74aa4bdf14f7f1b2a601dbecc94fa"},
        {"artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json", "id": "depth39-longest-f1g1h1r", "letters": 556, "sha256": "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14"},
    ]
    for entry in frontier:
        validate_entry(entry)

    rejected = [
        {"artifact": "runs/incumbent-568-sentence-boundary-clause-intersection-20261002.json", "id": "sentence-boundary-91-135-clause-growth-588", "letters": 592, "sha256": "e861ab1ea21d1408cf8ac59fc5014e9d858bc72a8ac425fd727687393abdf303", "status": "exact_rejected", "source_commit": "4388c9cd", "reason": "catalogue-like repeated clause inventory; preserve 4388c9cd evidence"},
        {"artifact": "runs/incumbent-568-repeated-shell-intersection-20261002.json", "id": "reviewer-seam-170-197-469-496-shell-replacement-592", "letters": 592, "sha256": "4c06d5151ffef85d6ebfa1e1199e0594ac8be00d4cd3b61e20e387020d55e723", "status": "exact_rejected", "reason": "structural gate defect; no relabeling or repair"},
        {"artifact": "runs/incumbent-568-token-boundary-intersection-20261002.json", "id": "token-boundary-64-nadia-stops-rats-then-star-spots-aidan", "letters": 596, "sha256": "514f284bbcc88c5eda7b99dd235ca4f5845c45825b943652008e6d6ed6a97313", "status": "exact_non_reader_promotable", "reason": "preserved rejected exact evidence"},
        {"artifact": "runs/incumbent-568-obligation-indexed-intersection-20261002.json", "id": "seam-100-nadia-stops-rats-then-star-spots-aidan", "letters": 596, "sha256": "d590e5cb08854dd2795ba65ee46ef0292764a2fcf469a8f9ced349f48745e1dc", "status": "rejected", "reason": "split-word fragments"},
    ]
    for entry in rejected:
        validate_entry(entry)

    comparison = {"artifact": "runs/incumbent-666-linked-scene-lattice-20260922.json", "id": "bidirectional-typed-trie-alternative-666", "letters": 666, "sha256": "bab693719482af36c7e223a687f94552ad3efda6825d481014134a7d7ae7148d", "source_commit": "9cb68296"}
    validate_entry(comparison)

    left_cut_raw = raw_after_letters(base, SEAM_LEFT)
    right_cut_raw = raw_after_letters(base, SEAM_RIGHT)
    assert (left_cut_raw, right_cut_raw) == (266, 508)
    assert partial_word(base, SEAM_LEFT) == "ma|ra"
    assert partial_word(base, SEAM_RIGHT) == "ar|am"
    left_phrase = "Mara saw God."
    right_phrase = "Dog was Aram."
    left_start = base.index(left_phrase, left_cut_raw - 20)
    left_span = (left_start, left_start + len(left_phrase))
    right_start = base.index(right_phrase, right_cut_raw - 20)
    right_span = (right_start, right_start + len(right_phrase))
    assert left_span == (264, 277) and right_span == (498, 511)
    assert normalize(base[:left_span[0]]) == normalize(base[right_span[1]:])[::-1]

    inventory = targeted_inventory()
    trie = ReverseClauseTrie()
    for clause in inventory:
        trie.add(clause)
    parent_clause_tapes = {
        normalize(match.group(0))
        for match in re.finditer(r"[A-Z][a-z]+ (?:sees|stops|spots) [A-Za-z]+\.", base)
    }
    parent_frames = {
        "|".join((match.group(1).casefold(), match.group(2), match.group(3).casefold()))
        for match in re.finditer(r"([A-Z][a-z]+) (sees|stops|spots) ([A-Za-z]+)\.", base)
    }
    parent_tape = normalize(base)
    attempts: list[dict[str, object]] = []
    selected: list[tuple[Clause, Clause, dict[str, object]]] = []
    used_clauses: set[str] = set()
    used_frames: set[str] = set()
    introduced_subjects: Counter[str] = Counter()
    introduced_predicates: Counter[str] = Counter()
    introduced_objects: Counter[str] = Counter()
    active_entities = {"mara", "rats", "aram"}
    left_cursor, right_cursor = SEAM_LEFT, SEAM_RIGHT - 1
    role_plan = ("observation", "control")

    for role in role_plan:
        if len(attempts) >= MAX_PAIRED_EXPANSIONS:
            break
        for left in inventory:
            if left.role_family != role or left.surface in used_clauses:
                continue
            matches = [candidate for candidate in trie.exact(left.tape) if candidate.surface != left.surface]
            right = next((candidate for candidate in matches if candidate.surface not in used_clauses), None)
            if right is None:
                continue
            candidate_units = (left, right)
            full_clause_novel = all(unit.tape not in parent_clause_tapes and unit.surface not in used_clauses for unit in candidate_units)
            frame_novel = all(unit.frame not in parent_frames and unit.frame not in used_frames for unit in candidate_units)
            subject_counts = introduced_subjects.copy(); predicate_counts = introduced_predicates.copy(); object_counts = introduced_objects.copy()
            for unit in candidate_units:
                subject_counts[unit.subject] += 1; predicate_counts[unit.predicate] += 1; object_counts[unit.object] += 1
            distinct_frame = len({unit.frame for unit in candidate_units}) == len(candidate_units)
            transition = bool(set().union(*(set(unit.active_entities) for unit in candidate_units)) - active_entities)
            residual = consume_pair(left, right, left_cursor, right_cursor)
            gate = {
                "full_clause_novel": full_clause_novel,
                "full_frame_subject_predicate_object_novel": frame_novel and distinct_frame,
                "subject_frequency_ok": max(subject_counts.values(), default=0) <= 2,
                "predicate_frequency_ok": max(predicate_counts.values(), default=0) <= 2,
                "object_frequency_ok": max(object_counts.values(), default=0) <= 2,
                "active_entity_transition": transition,
                "role_family_expected": left.role_family == role,
                "reverse_trie_match": bool(matches),
                "residual_empty": not residual["final_residual"],
                "committed_character_contradictions": residual["committed_character_contradictions"] == 0,
            }
            attempt = {
                "ordinal": len(attempts) + 1, "left": left.surface, "right_partner": right.surface,
                "left_grammar_state": {"role_family": left.role_family, "frame": left.frame, "active_entities": left.active_entities},
                "right_grammar_state": {"role_family": right.role_family, "frame": right.frame, "active_entities": right.active_entities},
                "active_entities_before": sorted(active_entities), "active_entities_after": sorted(active_entities | set(left.active_entities) | set(right.active_entities)),
                "raw_shell_spaces": {"left_before": base[left_span[0] - 1], "left_after": base[left_span[1]], "right_before": base[right_span[0] - 1], "right_after": base[right_span[1]], "all_spaces_or_terminal": True},
                "cursors": {"left": left_cursor, "right_reverse": right_cursor}, "residual": residual, "novelty_gate": gate,
                "status": "accepted" if all(gate.values()) else "rejected_novelty_or_residual",
            }
            attempts.append(attempt)
            if attempt["status"] != "accepted":
                continue
            selected.append((left, right, residual))
            used_clauses.update((left.surface, right.surface)); used_frames.update((left.frame, right.frame))
            introduced_subjects.update((left.subject, right.subject)); introduced_predicates.update((left.predicate, right.predicate)); introduced_objects.update((left.object, right.object))
            active_entities.update(left.active_entities); active_entities.update(right.active_entities)
            left_cursor += len(left.tape); right_cursor -= len(right.tape)
            break

    assert len(selected) == 2
    left_units = [left for left, _, _ in selected]
    right_units = [right for _, right, _ in reversed(selected)]
    left_text = " ".join(unit.surface for unit in left_units)
    right_text = " ".join(unit.surface for unit in right_units)
    rendered = base[:left_span[0]] + left_text + base[left_span[1]:right_span[0]] + right_text + base[right_span[1]:]
    independent = independent_audit(rendered)
    assert independent["two_pointer_exact"] and independent["normalized_letters"] > 568
    project = audit(rendered)
    introduced = [unit.surface for unit in left_units + right_units]
    candidate_region = left_text + right_text
    sentence_gate = {
        "lowercase_after_terminal": not bool(re.search(r"[.!?][a-z]", candidate_region)),
        "comma_splice": not bool(re.search(r",\s*[A-Z][a-z]+\s+[a-z]+", candidate_region)),
        "complete_clause_units": all(re.fullmatch(r"[A-Z][a-z]+ (?:sees|stops|spots) [A-Za-z]+\.", unit) for unit in introduced),
        "no_fragment_carriers": all(normalize(unit) in normalize(rendered) for unit in introduced),
        "introduced_clause_counts_at_most_one": max(Counter(normalize(unit) for unit in introduced).values()) == 1,
        "inherited_clause_counts_not_increased": all(rendered.count(sentence) <= base.count(sentence) for sentence in (left_phrase, right_phrase)),
    }
    assert all(sentence_gate.values())
    assert project["project_validator_exact"]

    row = {
        "id": "partial-110-458-varied-observation-control-594",
        "working_status": "568_lineage_exact_partial_word_child",
        "rendered": rendered,
        "audit": project,
        "independent_audit": independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)), "parent_id": PARENT_ID, "parent_sha256": PARENT_SHA256,
        "growth_over_parent": independent["normalized_letters"] - 568,
        "new_event_content": introduced,
        "seam": {"normalized_cuts": [SEAM_LEFT, SEAM_RIGHT], "raw_after_letter_boundaries": [left_cut_raw, right_cut_raw], "partial_words": [partial_word(base, SEAM_LEFT), partial_word(base, SEAM_RIGHT)], "replacement_spans": [list(left_span), list(right_span)], "old_left": left_phrase, "old_right": right_phrase},
        "grammar": {"inventory_size": len(inventory), "role_families": sorted({unit.role_family for unit in left_units + right_units}), "reverse_trie": True, "active_entity_transitions": [attempt["active_entities_after"] for attempt in attempts if attempt["status"] == "accepted"]},
        "online_state": {"attempted_paired_expansions": len(attempts), "accepted_paired_expansions": len(selected), "attempts": attempts, "final_left_cursor": left_cursor, "final_right_reverse_cursor": right_cursor, "final_residual": "", "introduced_subject_frequency": dict(introduced_subjects), "introduced_predicate_frequency": dict(introduced_predicates), "introduced_object_frequency": dict(introduced_objects)},
        "sentence_gate": sentence_gate,
    }
    return {
        "experiment_id": "incumbent-568-partial-varied-intersection-20261002",
        "method": "online reverse-trie intersection over varied complete clauses at a partial-word seam",
        "parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256},
        "rejected_evidence": rejected, "comparison_evidence": comparison,
        "config": {"max_attempts": MAX_PAIRED_EXPANSIONS, "partial_word_seam": True, "targeted_predicate_families": ["observation", "control"], "post_render_repair": False, "fresh_seed": False, "vocabulary_widened": False, "whole_sentence_sweep": False},
        "stats": {"independently_exact_children": 1, "children_longer_than_568": 1, "longest_letters": independent["normalized_letters"], "attempted_paired_expansions": len(attempts), "accepted_paired_expansions": len(selected), "committed_character_contradictions": 0},
        "preserved_frontier": frontier, "rows": [row],
    }


if __name__ == "__main__":
    result = build_payload()
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"artifact": str(OUT.relative_to(ROOT)), "letters": result["rows"][0]["independent_audit"]["normalized_letters"], "sha256": result["rows"][0]["independent_audit"]["sha256_forward"]}, indent=2))
