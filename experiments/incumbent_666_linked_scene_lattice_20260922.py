"""Seam-local bidirectional grammar/trie intersection on a 13-letter seam."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_666_boundary_discourse_linker_20260922 import FRONTIER, independent_audit, normalize, validate_frontier_entry
from experiments.incumbent_666_reciprocal_paired_production_20260922 import extract_frames


PARENT = ROOT / "runs" / "incumbent-666-context-aware-reciprocal-pair-20260922.json"
OUT = ROOT / "runs" / "incumbent-666-linked-scene-lattice-20260922.json"
PARENT_ID = "context-aware-reciprocal-pair-noel-sara-666"
PARENT_SHA256 = "a1b4ebaaba06893fdfa2a495676b887355e361266b2960a61818981e59e0da37"
NORMALIZED_LEFT = (127, 140)
NORMALIZED_RIGHT = (526, 539)
RAW_LEFT = (171, 188)
RAW_RIGHT = (719, 736)
TARGET_LETTERS = 13
MAX_PAIRED_EXPANSIONS = 8

# A small targeted grammar inventory, not a vocabulary product.  The
# stop/spot and saw/was relations already occur in the active lexical
# inventory; only clauses that can plausibly cross this seam are
# admitted to the tries.
CLAUSE_SPECS = (
    ("Aras", ("stops", "spots"), ("Aram", "Sara", "Nora")),
    ("Aram", ("stops", "spots"), ("Sara", "Mara")),
    ("Mara", ("stops", "spots"), ("Sara", "Aram", "Aras")),
    ("Sara", ("stops", "spots"), ("Aram", "Mara", "Nora")),
    ("Aidan", ("saw", "was"), ("Nadia", "Aidan")),
    ("Nadia", ("saw", "was"), ("Nadia", "Aidan")),
    ("Aron", ("stops", "spots"), ("Sara", "Aras")),
)
ALLOWED_TRANSITIONS = {
    ("left", "Aras", "Aras"): "boundary-continuity",
    ("right", "Aidan", "Mara"): "typed-paired-role-handoff",
}


@dataclass(frozen=True)
class Clause:
    subject: str
    predicate: str
    object: str

    @property
    def rendered(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}."

    @property
    def tape(self) -> str:
        return normalize(self.rendered)

    @property
    def frame(self) -> str:
        return f"{self.subject.lower()}|{self.predicate}"


@dataclass
class TrieNode:
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    terminals: list[Clause] = field(default_factory=list)


def inventory() -> tuple[Clause, ...]:
    rows: list[Clause] = []
    for subject, predicates, objects in CLAUSE_SPECS:
        for predicate in predicates:
            for object_ in objects:
                clause = Clause(subject, predicate, object_)
                if len(clause.tape) == TARGET_LETTERS:
                    rows.append(clause)
    return tuple(rows)


def add(root: TrieNode, tape: str, clause: Clause) -> None:
    node = root
    for char in tape:
        node = node.children.setdefault(char, TrieNode())
    node.terminals.append(clause)


def build_tries(left_allowed: tuple[Clause, ...] | None = None, right_allowed: tuple[Clause, ...] | None = None) -> tuple[TrieNode, TrieNode, tuple[Clause, ...]]:
    left_root, right_reverse_root = TrieNode(), TrieNode()
    clauses = inventory()
    left_allowed = clauses if left_allowed is None else left_allowed
    right_allowed = clauses if right_allowed is None else right_allowed
    for clause in left_allowed:
        add(left_root, clause.tape, clause)
    for clause in right_allowed:
        add(right_reverse_root, clause.tape[::-1], clause)
    return left_root, right_reverse_root, tuple(sorted(set(left_allowed) | set(right_allowed), key=lambda clause: clause.rendered))


def parent_clauses(text: str) -> set[str]:
    return {part.strip().lower() + "." for part in re.split(r"[.!?;]+", text) if part.strip()}


def transition(side: str, before: str, subject: str) -> tuple[bool, str]:
    if before == subject:
        return True, "boundary-continuity"
    label = ALLOWED_TRANSITIONS.get((side, before, subject))
    return (label is not None, label or "rejected-subject-transition")


def assemble(parent: str, left: Clause, right: Clause) -> tuple[str, dict[str, object]]:
    left_text = left.rendered + " "
    right_text = right.rendered + " "
    assert parent[RAW_LEFT[0] - 1] == parent[RAW_RIGHT[0] - 1] == " "
    assert parent[RAW_LEFT[1] - 1] == parent[RAW_RIGHT[1] - 1] == " "
    rendered = parent[: RAW_LEFT[0]] + left_text + parent[RAW_LEFT[1] : RAW_RIGHT[0]] + right_text + parent[RAW_RIGHT[1] :]
    left_end = RAW_LEFT[0] + len(left_text)
    right_start = RAW_RIGHT[0] + (len(left_text) - (RAW_LEFT[1] - RAW_LEFT[0]))
    right_end = right_start + len(right_text)
    left_leading_preserved = rendered[RAW_LEFT[0] - 1] == " " and rendered[RAW_LEFT[0]] == left.subject[0]
    right_leading_preserved = rendered[right_start - 1] == " " and rendered[right_start] == right.subject[0]
    left_trailing_preserved = rendered[left_end - 1] == " " and rendered[left_end] == parent[RAW_LEFT[1]]
    right_trailing_preserved = rendered[right_end - 1] == " " and rendered[right_end] == parent[RAW_RIGHT[1]]
    spacing = {
        "left_leading_space_owner": repr(parent[RAW_LEFT[0] - 1]),
        "right_leading_space_owner": repr(parent[RAW_RIGHT[0] - 1]),
        "left_trailing_space_owner": repr(parent[RAW_LEFT[1] - 1]),
        "right_trailing_space_owner": repr(parent[RAW_RIGHT[1] - 1]),
        "left_prefix_excerpt": rendered[RAW_LEFT[0] - 3 : RAW_LEFT[0] + 20],
        "right_prefix_excerpt": rendered[right_start - 3 : right_start + 20],
        "left_leading_preserved": left_leading_preserved,
        "right_leading_preserved": right_leading_preserved,
        "left_trailing_preserved": left_trailing_preserved,
        "right_trailing_preserved": right_trailing_preserved,
    }
    spacing["spacing_shell_preserved"] = all(value for key, value in spacing.items() if key.endswith("preserved"))
    return rendered, spacing


def intersect(parent: str) -> dict[str, object]:
    all_clauses = inventory()
    left_grammar = tuple(clause for clause in all_clauses if transition("left", "Aras", clause.subject)[0])
    right_grammar = tuple(clause for clause in all_clauses if transition("right", "Aidan", clause.subject)[0])
    left_root, right_root, clauses = build_tries(left_grammar, right_grammar)
    pframes = extract_frames(parent)
    pclauses = parent_clauses(parent)
    expansions = 0
    states: list[dict[str, object]] = []
    closures: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []

    def descendants(node: TrieNode) -> tuple[Clause, ...]:
        found = list(node.terminals)
        for child in node.children.values():
            found.extend(descendants(child))
        return tuple(sorted(set(found), key=lambda clause: clause.rendered))

    def walk(
        left_node: TrieNode,
        right_node: TrieNode,
        cursor: int,
        left_tape: str,
        right_reverse_tape: str,
        left_before: str,
        right_before: str,
        used_frames: frozenset[str],
        used_clauses: frozenset[str],
    ) -> None:
        nonlocal expansions
        keys = sorted(set(left_node.children) & set(right_node.children))
        if not keys:
            states.append({
                "cursor": cursor,
                "left_cursor": cursor,
                "right_cursor": cursor,
                "right_reverse_cursor": cursor,
                "left_residual": TARGET_LETTERS - cursor,
                "right_residual": TARGET_LETTERS - cursor,
                "right_reverse_residual": TARGET_LETTERS - cursor,
                "active_entities": {"left": left_before, "right": right_before},
                "used_frames": sorted(used_frames),
                "used_clauses": sorted(used_clauses),
                "owner": "grammar-trie-intersection",
                "reason": "no_shared_next_character",
            })
            return
        for char in keys:
            left_child = left_node.children[char]
            right_child = right_node.children[char]
            next_left = left_tape + char
            next_right_reverse = right_reverse_tape + char
            left_frontier = descendants(left_child)
            right_frontier = descendants(right_child)
            frontier_frames = {clause.frame for clause in left_frontier + right_frontier}
            frontier_clauses = {clause.rendered.lower() for clause in left_frontier + right_frontier}
            state = {
                "cursor": cursor + 1,
                "left_cursor": cursor + 1,
                "right_cursor": cursor + 1,
                "right_reverse_cursor": cursor + 1,
                "left_emitted": char,
                "right_reverse_emitted": char,
                "left_residual": TARGET_LETTERS - cursor - 1,
                "right_residual": TARGET_LETTERS - cursor - 1,
                "right_reverse_residual": TARGET_LETTERS - cursor - 1,
                "left_residual_tape": next_left,
                "right_reverse_residual_tape": next_right_reverse,
                "active_left_entity": left_before,
                "active_right_entity": right_before,
                "active_entities": {"left": left_before, "right": right_before},
                "frontier_frames": sorted(frontier_frames),
                "frontier_clauses": sorted(frontier_clauses),
                "parent_frame_reuse": sorted(frontier_frames & pframes),
                "parent_clause_reuse": sorted(frontier_clauses & pclauses),
                "novel_frontier_frames": sorted(frontier_frames - pframes - set(used_frames)),
                "novel_frontier_clauses": sorted(frontier_clauses - pclauses - set(used_clauses)),
                "used_frames": sorted(used_frames),
                "used_clauses": sorted(used_clauses),
                "owner": "grammar-trie-intersection",
            }
            states.append(state)
            if cursor + 1 < TARGET_LETTERS:
                walk(left_child, right_child, cursor + 1, next_left, next_right_reverse, left_before, right_before, used_frames, used_clauses)
                continue
            terminal_pairs = [(left_clause, right_clause) for left_clause in left_child.terminals for right_clause in right_child.terminals]
            terminal_pairs.sort(key=lambda pair: (
                not transition("left", left_before, pair[0].subject)[0],
                not transition("right", right_before, pair[1].subject)[0],
                bool({pair[0].frame, pair[1].frame} & pframes),
                bool({pair[0].rendered.lower(), pair[1].rendered.lower()} & pclauses),
                pair[0].rendered,
                pair[1].rendered,
            ))
            for left_clause, right_clause in terminal_pairs:
                if expansions >= MAX_PAIRED_EXPANSIONS:
                    return
                expansions += 1
                left_ok, left_transition = transition("left", left_before, left_clause.subject)
                right_ok, right_transition = transition("right", right_before, right_clause.subject)
                frames = frozenset({left_clause.frame, right_clause.frame})
                clause_keys = frozenset({left_clause.rendered.lower(), right_clause.rendered.lower()})
                novelty = not (
                    (frames & pframes)
                    or (clause_keys & pclauses)
                    or (frames & used_frames)
                    or (clause_keys & used_clauses)
                )
                result = {
                    "left_clause": left_clause.rendered,
                    "right_clause": right_clause.rendered,
                    "left_frame": left_clause.frame,
                    "right_frame": right_clause.frame,
                    "left_transition": left_transition,
                    "right_transition": right_transition,
                    "left_active_entity_before": left_before,
                    "left_active_entity_after": left_clause.object,
                    "right_active_entity_before": right_before,
                    "right_active_entity_after": right_clause.object,
                    "used_frames": sorted(frames | used_frames),
                    "used_clauses": sorted(clause_keys | used_clauses),
                    "parent_frame_reuse": sorted(frames & pframes),
                    "parent_clause_reuse": sorted(clause_keys & pclauses),
                    "cursor": [TARGET_LETTERS, TARGET_LETTERS],
                    "left_cursor": TARGET_LETTERS,
                    "right_reverse_cursor": TARGET_LETTERS,
                    "residuals": {"left": "", "right_reverse": ""},
                    "incremental_novelty": {
                        "parent_frames_clear": not bool(frames & pframes),
                        "parent_clauses_clear": not bool(clause_keys & pclauses),
                        "used_frames_clear": not bool(frames & used_frames),
                        "used_clauses_clear": not bool(clause_keys & used_clauses),
                    },
                }
                if not left_ok or not right_ok or not novelty:
                    result["reason"] = "subject-transition-or-incremental-novelty-rejection"
                    rejected.append(result)
                    continue
                result["accepted"] = True
                closures.append(result)

    walk(left_root, right_root, 0, "", "", "Aras", "Aidan", frozenset(), frozenset())
    return {"expansions": expansions, "states": states, "closures": closures, "rejected": rejected, "inventory_size": len(clauses), "left_grammar_size": len(left_grammar), "right_grammar_size": len(right_grammar), "parent_frame_count": len(pframes), "parent_clause_count": len(pclauses), "character_state_count": len(states)}


def build_payload() -> dict[str, object]:
    payload = json.loads(PARENT.read_text())
    parent = next(row for row in payload["rows"] if row["id"] == PARENT_ID)
    rendered = str(parent["rendered"])
    parent_audit = independent_audit(rendered)
    assert parent_audit["normalized_letters"] == 666 and parent_audit["two_pointer_exact"] and parent_audit["sha256_forward"] == PARENT_SHA256
    assert parent["promotion_status"]["promoted"] is True
    for entry in FRONTIER:
        validate_frontier_entry(entry)
    tape = normalize(rendered)
    assert tape[NORMALIZED_LEFT[0] : NORMALIZED_LEFT[1]] == tape[NORMALIZED_RIGHT[0] : NORMALIZED_RIGHT[1]][::-1]
    result = intersect(rendered)
    assert result["expansions"] <= MAX_PAIRED_EXPANSIONS
    assert result["closures"]
    closure = result["closures"][0]
    left_clause = Clause(*re.match(r"([A-Za-z]+) ([a-z]+) ([A-Za-z]+)\.", closure["left_clause"]).groups())
    right_clause = Clause(*re.match(r"([A-Za-z]+) ([a-z]+) ([A-Za-z]+)\.", closure["right_clause"]).groups())
    candidate, spacing = assemble(rendered, left_clause, right_clause)
    audit = independent_audit(candidate)
    assert audit["normalized_letters"] == 666 and audit["two_pointer_exact"]
    assert spacing["spacing_shell_preserved"]
    row = {
        "id": "bidirectional-typed-trie-alternative-666",
        "working_status": "exact_comparison_pending_full_text_review",
        "promotion_status": {"promoted": False, "status": "comparison_pending_full_text_review", "reason": "exact parent-novel frame/clauses from bounded seam-local grammar; working incumbent remains 568"},
        "rendered": candidate,
        "independent_audit": audit,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": 0,
        "alternate_seam": {"normalized_windows": {"left": list(NORMALIZED_LEFT), "right": list(NORMALIZED_RIGHT)}, "raw_windows": {"left": list(RAW_LEFT), "right": list(RAW_RIGHT)}, "old_left": rendered[RAW_LEFT[0] : RAW_LEFT[1]], "old_right": rendered[RAW_RIGHT[0] : RAW_RIGHT[1]]},
        "typed_trie_intersection": {"predicate_inventory": sorted({clause.predicate for clause in inventory()}), "targeted_subjects": sorted({clause.subject for clause in inventory()}), "max_paired_expansions": MAX_PAIRED_EXPANSIONS, "result": result, "selected_closure": closure, "selected_candidate": {"left": left_clause.rendered, "right": right_clause.rendered, "normalized_letters_per_side": TARGET_LETTERS, "sha256": audit["sha256_forward"], "spacing": spacing, "both_active_entities": {"left_before": "Aras", "left_after": left_clause.object, "right_before": "Aidan", "right_after": right_clause.object}}, "admission": {"accepted": True, "exact_child_saved": True, "independently_exact": True, "new_event_content": True}},
        "next_operator": "full-text review of exact 666 comparison; if rejected, change to a different actual seam while preserving this exact child and the 568/560/558/556 frontier",
        "provenance": "obligation-indexed forward/reverse clause-trie intersection with incremental parent novelty and typed subject transitions",
    }
    return {"experiment_id": "incumbent-666-linked-scene-lattice-20260922", "method": "bounded bidirectional typed clause-trie intersection on alternate 13-letter seam", "active_frontier_parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 666, "sha256": PARENT_SHA256}, "working_incumbent": {"artifact": "runs/incumbent-560-outer-causal-scene-20261002.json", "id": "outer-causal-scene-568-working-incumbent", "letters": 568, "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"}, "preserved_frontier": list(FRONTIER), "rows": [row], "next_operator": row["next_operator"]}


def main() -> None:
    result = build_payload()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    row = result["rows"][0]
    selected = row["typed_trie_intersection"]["selected_candidate"]
    print({"id": row["id"], "left": selected["left"], "right": selected["right"], "sha256": selected["sha256"], "expansions": row["typed_trie_intersection"]["result"]["expansions"]})


if __name__ == "__main__":
    main()
