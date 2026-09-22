"""Sentence-boundary-only clause expansion from the authoritative 568 parent."""
from __future__ import annotations

import itertools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit
from experiments.incumbent_568_obligation_indexed_intersection_20261002 import (
    PARENT,
    PARENT_ID,
    PARENT_SHA256,
    grammar_inventory,
    independent_audit,
    normalize,
    raw_boundary_after_letters,
)

OUT = ROOT / "runs" / "incumbent-568-sentence-boundary-clause-intersection-20261002.json"
MAX_ATTEMPTS = 8
OLD_WINDOW_LETTERS = 44

REJECTED_EVIDENCE = [
    {"artifact": "runs/incumbent-568-repeated-shell-intersection-20261002.json", "id": "reviewer-seam-170-197-469-496-shell-replacement-592", "letters": 592, "sha256": "4c06d5151ffef85d6ebfa1e1199e0594ac8be00d4cd3b61e20e387020d55e723", "status": "exact_rejected", "reason": "structural gate defect in prior operator; preserve without relabeling or repair"},
    {"artifact": "runs/incumbent-568-token-boundary-intersection-20261002.json", "id": "token-boundary-64-nadia-stops-rats-then-star-spots-aidan", "letters": 596, "sha256": "514f284bbcc88c5eda7b99dd235ca4f5845c45825b943652008e6d6ed6a97313", "status": "exact_non_reader_promotable", "reason": "preserved prior exact evidence"},
    {"artifact": "runs/incumbent-568-obligation-indexed-intersection-20261002.json", "id": "seam-100-nadia-stops-rats-then-star-spots-aidan", "letters": 596, "sha256": "d590e5cb08854dd2795ba65ee46ef0292764a2fcf469a8f9ced349f48745e1dc", "status": "rejected", "reason": "split-word fragments"},
]


@dataclass(frozen=True)
class ClausePair:
    left: object
    right: object

    @property
    def left_tape(self) -> str:
        return normalize(self.left.surface)

    @property
    def right_tape(self) -> str:
        return normalize(self.right.surface)


class ReverseClauseTrie:
    def __init__(self) -> None:
        self.children: dict[str, "ReverseClauseTrie"] = {}
        self.terminals: list[object] = []

    def add(self, clause: object) -> None:
        node = self
        for char in normalize(clause.surface)[::-1]:
            node = node.children.setdefault(char, ReverseClauseTrie())
        node.terminals.append(clause)

    def exact(self, tape: str) -> tuple[object, ...]:
        node = self
        for char in tape:
            node = node.children.get(char)
            if node is None:
                return ()
        return tuple(node.terminals)


def raw_after(text: str, letters: int) -> int:
    return raw_boundary_after_letters(text, letters)


def sentence_start(text: str, letters: int) -> int:
    index = raw_after(text, letters)
    while index < len(text) and not text[index].isalpha():
        index += 1
    return index


def sentence_end(text: str, letters: int) -> int:
    index = raw_after(text, letters)
    while index < len(text) and text[index] != ".":
        index += 1
    assert index < len(text) and text[index] == "."
    index += 1
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def consume_clause_sequence(pairs: tuple[ClausePair, ...], left_cursor: int, right_cursor: int) -> dict[str, object]:
    left = " ".join(pair.left.surface for pair in pairs) + " "
    right = " ".join(pair.right.surface for pair in reversed(pairs)) + " "
    left_tape = normalize(left)
    right_reverse = normalize(right)[::-1]
    left_residual, right_residual = left_tape, right_reverse
    trace: list[dict[str, object]] = []
    contradictions = 0
    for offset, (left_char, right_char) in enumerate(zip(left_tape, right_reverse)):
        matched = left_char == right_char
        trace.append({"left_cursor": left_cursor + offset, "right_reverse_cursor": right_cursor - offset - 1, "left_emitted": left_char, "right_reverse_emitted": right_char, "matched": matched})
        if not matched:
            contradictions += 1
            break
        left_residual, right_residual = left_residual[1:], right_residual[1:]
    return {"left_emission": left_tape, "right_reverse_obligation": right_reverse, "left_residual": left_residual, "right_reverse_residual": right_residual, "final_residual": left_residual, "left_cursor_after": left_cursor + len(left_tape), "right_reverse_cursor_after": right_cursor - len(right_reverse), "committed_character_contradictions": contradictions, "trace": trace}


def complete_clause_gate(text: str, clauses: tuple[str, ...]) -> dict[str, object]:
    units = [unit.strip() for unit in text.split(".") if unit.strip()]
    terminal_ok = all(unit and unit[0].isupper() for unit in units)
    lowercase_after_terminal = bool(re.search(r"[.!?][a-z]", text))
    comma_splice = bool(re.search(r",\s+[A-Z][a-z]+\s+(?:sees|stops|spots|maps|delivers|rewards|saw|was|won)\b", text))
    fragment = not all(re.fullmatch(r"[A-Z][a-z]+\s+(?:sees|stops|spots|maps|delivers|rewards)\s+[A-Za-z]+\.", clause) for clause in clauses)
    return {"lowercase_after_terminal": lowercase_after_terminal, "comma_splice": comma_splice, "fragment": fragment, "all_emitted_units_complete": terminal_ok and not fragment, "clauses_emitted_once": all(text.count(clause) == 1 for clause in clauses)}


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    base = str(parent["rendered"])
    base_tape = normalize(base)
    assert len(base_tape) == 568 and base_tape == base_tape[::-1]
    assert independent_audit(base)["sha256_forward"] == PARENT_SHA256

    frontier = [
        {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256},
        {"artifact": "runs/incumbent-550-central-event-bridge-20261002.json", "id": "central-distinct-events-560", "letters": 560, "sha256": "b5f98bfb0b44b31d8cbf78727672a74b588980e1fc8f1ff522a2c4ad1d800ccc"},
        {"artifact": "runs/incumbent-550-typed-center-product-20261002.json", "id": "typed-center-25", "letters": 558, "sha256": "29470b5ab408c402e8796530123357fea6a74aa4bdf14f7f1b2a601dbecc94fa"},
        {"artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json", "id": "depth39-longest-f1g1h1r", "letters": 556, "sha256": "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14"},
    ]
    for entry in frontier:
        payload = json.loads((ROOT / entry["artifact"]).read_text())
        row = next(r for r in payload["rows"] if r["id"] == entry["id"])
        checked = independent_audit(str(row["rendered"]))
        assert checked["normalized_letters"] == entry["letters"] and checked["two_pointer_exact"] and checked["sha256_forward"] == entry["sha256"]

    comparison = {"artifact": "runs/incumbent-666-linked-scene-lattice-20260922.json", "id": "bidirectional-typed-trie-alternative-666", "letters": 666, "sha256": "bab693719482af36c7e223a687f94552ad3efda6825d481014134a7d7ae7148d", "source_commit": "9cb68296"}
    cpayload = json.loads((ROOT / comparison["artifact"]).read_text())
    crow = next(r for r in cpayload["rows"] if r["id"] == comparison["id"])
    caudit = independent_audit(str(crow["rendered"]))
    assert caudit["normalized_letters"] == 666 and caudit["two_pointer_exact"] and caudit["sha256_forward"] == comparison["sha256"]

    requested = ((91, 135), (433, 477))
    raw_after_spans = tuple((raw_after(base, a), raw_after(base, b)) for a, b in requested)
    boundary_spans = ((sentence_start(base, 91), sentence_end(base, 135)), (sentence_start(base, 433), sentence_end(base, 477)))
    assert raw_after_spans == ((119, 178), (598, 660))
    assert boundary_spans == ((121, 180), (600, 662))
    left_start, left_end = boundary_spans[0]
    right_start, right_end = boundary_spans[1]
    old_left, old_right = base[left_start:left_end], base[right_start:right_end]
    assert old_left == "Aidan delivers maps. Mara stops rats. A tub? He maps Nora. "
    assert old_right == "Aron, spam. Eh, but a star spots Aram. Spam's reviled, Nadia. "
    assert normalize(old_left) == normalize(old_right)[::-1]

    clauses = grammar_inventory()
    trie = ReverseClauseTrie()
    for clause in clauses:
        trie.add(clause)
    parent_tape = normalize(base)
    directed_pairs: list[ClausePair] = []
    for left in clauses:
        words = re.findall(r"[A-Za-z]+", left.surface)
        if words[0].casefold() == words[-1].casefold():
            continue
        for right in trie.exact(normalize(left.surface)):
            right_words = re.findall(r"[A-Za-z]+", right.surface)
            if left.surface != right.surface and right_words[0].casefold() != right_words[-1].casefold():
                directed_pairs.append(ClausePair(left, right))
    unique_pairs = {(pair.left.surface, pair.right.surface): pair for pair in directed_pairs}
    directed_pairs = tuple(unique_pairs.values())
    attempts: list[dict[str, object]] = []
    chosen = None
    for combo in itertools.combinations(directed_pairs, 4):
        if len(attempts) >= MAX_ATTEMPTS:
            break
        left_clauses = tuple(pair.left.surface for pair in combo)
        right_clauses = tuple(pair.right.surface for pair in reversed(combo))
        all_clauses = left_clauses + right_clauses
        if len(set(all_clauses)) != len(all_clauses):
            continue
        frame_signatures = [tuple(re.findall(r"[A-Za-z]+", clause.casefold())[:2]) for clause in all_clauses]
        introduced_frame_repetition = max(frame_signatures.count(signature) for signature in set(frame_signatures)) > 2
        if introduced_frame_repetition:
            continue
        if any(normalize(clause) in parent_tape for clause in all_clauses):
            continue
        left_rendered = " ".join(left_clauses) + " "
        right_rendered = " ".join(right_clauses) + " "
        if len(normalize(left_rendered)) <= OLD_WINDOW_LETTERS:
            continue
        residual = consume_clause_sequence(combo, 91, 477)
        gate = complete_clause_gate(left_rendered, left_clauses) | {"right": complete_clause_gate(right_rendered, right_clauses)}
        attempt = {"ordinal": len(attempts) + 1, "left_clauses": left_clauses, "right_clauses": right_clauses, "grammar_state": {"frames": [pair.left.frame for pair in combo] + [pair.right.frame for pair in reversed(combo)], "active_entities": sorted({entity for pair in combo for entity in pair.left.active_entities + pair.right.active_entities}), "parent_clause_novel": True, "introduced_frame_repetition": introduced_frame_repetition}, "residual": residual, "gate": gate, "status": "accepted" if not residual["final_residual"] and not residual["committed_character_contradictions"] and gate["all_emitted_units_complete"] and gate["right"]["all_emitted_units_complete"] else "rejected_gate"}
        attempts.append(attempt)
        if attempt["status"] == "accepted":
            chosen = (combo, residual, left_rendered, right_rendered)
            break
    assert chosen is not None
    combo, residual, left_rendered, right_rendered = chosen
    rendered = base[:left_start] + left_rendered + base[left_end:right_start] + right_rendered + base[right_end:]
    project = audit(rendered)
    independent = independent_audit(rendered)
    assert independent["two_pointer_exact"] and independent["sha_equal"] and independent["normalized_letters"] > 568

    emitted = tuple(pair.left.surface for pair in combo) + tuple(pair.right.surface for pair in reversed(combo))
    full_gate_left = complete_clause_gate(left_rendered, tuple(pair.left.surface for pair in combo))
    full_gate_right = complete_clause_gate(right_rendered, tuple(pair.right.surface for pair in reversed(combo)))
    frame_signatures = [tuple(re.findall(r"[A-Za-z]+", clause.casefold())[:2]) for clause in emitted]
    full_gate = {"lowercase_after_terminal": full_gate_left["lowercase_after_terminal"] or full_gate_right["lowercase_after_terminal"], "comma_splice": full_gate_left["comma_splice"] or full_gate_right["comma_splice"], "fragment": full_gate_left["fragment"] or full_gate_right["fragment"], "all_emitted_units_complete": full_gate_left["all_emitted_units_complete"] and full_gate_right["all_emitted_units_complete"], "inserted_unit_duplicated": not all(rendered.count(unit) == 1 for unit in emitted), "introduced_frame_repetition": max(frame_signatures.count(signature) for signature in set(frame_signatures)) > 2, "worsened_inherited_repetition": any(rendered.count(phrase) > base.count(phrase) for phrase in ("Mara stops rats.", "A tub?", "Eh, but a star spots Aram.", "Spam's reviled")), "status": "passed structural full-text gate; readability remains unpromoted"}
    assert not full_gate["lowercase_after_terminal"] and not full_gate["comma_splice"] and not full_gate["fragment"] and full_gate["all_emitted_units_complete"] and not full_gate["inserted_unit_duplicated"] and not full_gate["introduced_frame_repetition"] and not full_gate["worsened_inherited_repetition"]

    row = {"id": "sentence-boundary-91-135-clause-growth-588", "working_status": "568_lineage_exact_sentence_boundary_child", "rendered": rendered, "audit": project, "independent_audit": independent, "parent_artifact": str(PARENT.relative_to(ROOT)), "parent_id": PARENT_ID, "parent_sha256": PARENT_SHA256, "growth_over_parent": independent["normalized_letters"] - 568, "new_event_content": list(emitted), "reviewer_seam": {"requested_normalized_windows": [list(x) for x in requested], "raw_after_letter_boundaries": [list(x) for x in raw_after_spans], "sentence_boundary_spans": [list(x) for x in boundary_spans], "old_left": old_left, "old_right": old_right}, "live_bidirectional_residual": residual, "grammar_novelty": {"active_entities": sorted({entity for pair in combo for entity in pair.left.active_entities + pair.right.active_entities}), "parent_clause_reuse": False, "complete_clause_count_per_side": 4}, "full_text_gate": full_gate, "attempts": attempts, "provenance": "authoritative 568 parent; complete sentence spans replaced by variable-length reverse-trie clause sequence; no partial-word carrier or post-render repair"}
    return {"experiment_id": "incumbent-568-sentence-boundary-clause-intersection-20261002", "method": "variable-length complete-clause reverse-trie intersection at sentence boundaries", "parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256}, "rejected_evidence": REJECTED_EVIDENCE, "comparison_evidence": comparison, "config": {"max_attempts": MAX_ATTEMPTS, "requested_normalized_windows": [list(x) for x in requested], "complete_sentence_boundaries_only": True, "variable_length": True, "post_render_repair": False, "fresh_seed": False, "vocabulary_widened": False}, "stats": {"independently_exact_children": 1, "children_longer_than_568": 1, "longest_letters": independent["normalized_letters"], "attempted_paired_expansions": len(attempts), "committed_character_contradictions": 0, "complete_clauses_per_side": 4}, "preserved_frontier": frontier, "rows": [row], "next_operator": "retain this exact sentence-boundary child as non-promoted evidence; if reviewers reject it, switch seam without post-render repair"}


def main() -> None:
    payload = build_payload()
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
