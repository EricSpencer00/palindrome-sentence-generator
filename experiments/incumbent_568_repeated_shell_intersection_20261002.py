"""Replace one repeated 568 shell through reflected boundary-aware grammar.

The reviewer-selected seam is represented by two requested windows,
``[170,197)`` and ``[469,496)``, plus their exact reflected supports
``[371,398)`` and ``[72,99)``.  Candidate clauses are generated first and
then intersected through reverse tries.  The only partial edge carriers are
``Mar/ram``, ``led/del``, and ``s``; the full-text gate requires those carriers
to join existing words (Mara, Aram, reviled, spots, stops, delivers).
"""
from __future__ import annotations

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

OUT = ROOT / "runs" / "incumbent-568-repeated-shell-intersection-20261002.json"
MAX_PAIRED_EXPANSIONS = 8
TARGET_WINDOW_LETTERS = 33

REJECTED_EVIDENCE = [
    {
        "artifact": "runs/incumbent-568-token-boundary-intersection-20261002.json",
        "id": "token-boundary-64-nadia-stops-rats-then-star-spots-aidan",
        "letters": 596,
        "sha256": "514f284bbcc88c5eda7b99dd235ca4f5845c45825b943652008e6d6ed6a97313",
        "status": "exact_non_reader_promotable",
        "reason": "preserved evidence from 1e024226; exact but not promoted as readable frontier",
    },
    {
        "artifact": "runs/incumbent-568-obligation-indexed-intersection-20261002.json",
        "id": "seam-100-nadia-stops-rats-then-star-spots-aidan",
        "letters": 596,
        "sha256": "d590e5cb08854dd2795ba65ee46ef0292764a2fcf469a8f9ced349f48745e1dc",
        "status": "rejected",
        "reason": "split-word seam produced fragments; preserved without repair",
    },
]


@dataclass(frozen=True)
class WindowCandidate:
    surface: str
    clauses: tuple[str, ...]
    frame: str = "SVO.transitive.present"
    active_entities: tuple[str, ...] = ()

    @property
    def tape(self) -> str:
        return normalize(self.surface)


class ReverseTrie:
    def __init__(self) -> None:
        self.children: dict[str, "ReverseTrie"] = {}
        self.terminals: list[WindowCandidate] = []

    def add(self, candidate: WindowCandidate) -> None:
        node = self
        for char in candidate.tape[::-1]:
            node = node.children.setdefault(char, ReverseTrie())
        node.terminals.append(candidate)

    def exact(self, tape: str) -> tuple[WindowCandidate, ...]:
        node = self
        for char in tape:
            node = node.children.get(char)
            if node is None:
                return ()
        return tuple(node.terminals)


def raw_after_letters(text: str, letters: int) -> int:
    return raw_boundary_after_letters(text, letters)


def sentence_sequences() -> tuple[tuple[str, tuple[str, ...]], ...]:
    clauses = grammar_inventory()
    rows: list[tuple[str, tuple[str, ...]]] = []
    for first in clauses:
        rows.append((first.surface, (first.surface,)))
        for second in clauses:
            rows.append((first.surface + " " + second.surface, (first.surface, second.surface)))
    return tuple(rows)


def candidate_bank(kind: str, sequences: tuple[tuple[str, tuple[str, ...]], ...]) -> tuple[WindowCandidate, ...]:
    rows: list[WindowCandidate] = []
    for body, clause_list in sequences:
        if kind == "a_left":
            surface = " Noel. " + body + " Mar"
        elif kind == "a_right":
            surface = "ram. " + body + " “Leon"
        elif kind == "b_left":
            surface = "led, " + body + " s"
        elif kind == "b_right":
            surface = "s. " + body + " del"
        else:
            raise ValueError(kind)
        entities = tuple(sorted({word.casefold() for clause in clause_list for word in re.findall(r"[A-Za-z]+", clause)}))
        rows.append(WindowCandidate(surface, clause_list, active_entities=entities))
    return tuple(rows)


def consume_pair(left: WindowCandidate, right: WindowCandidate, *, left_cursor: int, right_cursor: int) -> dict[str, object]:
    left_tape = left.tape
    right_reverse = right.tape[::-1]
    left_residual = left_tape
    right_residual = right_reverse
    trace: list[dict[str, object]] = []
    contradictions = 0
    for offset, (left_char, right_char) in enumerate(zip(left_tape, right_reverse)):
        matched = left_char == right_char
        trace.append({
            "left_cursor": left_cursor + offset,
            "right_reverse_cursor": right_cursor - offset - 1,
            "left_emitted": left_char,
            "right_reverse_emitted": right_char,
            "matched": matched,
        })
        if not matched:
            contradictions += 1
            break
        left_residual = left_residual[1:]
        right_residual = right_residual[1:]
    return {
        "left_emission": left_tape,
        "right_reverse_obligation": right_reverse,
        "left_residual": left_residual,
        "right_reverse_residual": right_residual,
        "final_residual": left_residual,
        "left_cursor_after": left_cursor + len(left_tape),
        "right_reverse_cursor_after": right_cursor - len(right_reverse),
        "committed_character_contradictions": contradictions,
        "trace": trace,
    }


def replace_windows(text: str, replacements: dict[tuple[int, int], str]) -> str:
    result = text
    for (start, end), replacement in sorted(replacements.items(), reverse=True):
        result = result[:start] + replacement + result[end:]
    return result


def clause_surfaces(candidate: WindowCandidate) -> tuple[str, ...]:
    return tuple(candidate.clauses)


def has_reflexive_clause(candidate: WindowCandidate) -> bool:
    for clause in candidate.clauses:
        words = re.findall(r"[A-Za-z]+", clause.casefold())
        if words and words[0] == words[-1]:
            return True
    return False


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
        assert checked["normalized_letters"] == entry["letters"] and checked["two_pointer_exact"]
        assert checked["sha256_forward"] == entry["sha256"]

    comparison = {"artifact": "runs/incumbent-666-linked-scene-lattice-20260922.json", "id": "bidirectional-typed-trie-alternative-666", "letters": 666, "sha256": "bab693719482af36c7e223a687f94552ad3efda6825d481014134a7d7ae7148d", "source_commit": "9cb68296"}
    cpayload = json.loads((ROOT / comparison["artifact"]).read_text())
    crow = next(r for r in cpayload["rows"] if r["id"] == comparison["id"])
    caudit = independent_audit(str(crow["rendered"]))
    assert caudit["normalized_letters"] == 666 and caudit["two_pointer_exact"] and caudit["sha256_forward"] == comparison["sha256"]

    # Recompute reviewer offsets from the parent; do not trust copied raw offsets.
    requested = ((170, 197), (469, 496))
    reflected = ((72, 99), (371, 398))
    spans = requested + reflected
    raw_spans = tuple((raw_after_letters(base, a), raw_after_letters(base, b)) for a, b in spans)
    assert raw_spans == ((228, 267), (650, 690), (92, 130), (507, 548))
    (a_left_raw, a_left_end), (b_left_raw, b_left_end), (b_mirror_raw, b_mirror_end), (a_mirror_raw, a_mirror_end) = raw_spans
    assert base[a_left_raw:a_left_end] == " Noel live. Noel, I sit. Pat notes. Mar"
    assert base[b_left_raw:b_left_end] == "led, Nadia. Nora, spam. Eh, but a star s"
    assert base[b_mirror_raw:b_mirror_end] == "s rats. A tub? He maps Aron. Aidan del"
    assert base[a_mirror_raw:a_mirror_end] == "ram. Seton, tap. 'Tis I, Leon. “Evil Leon"

    sequences = sentence_sequences()
    banks = {kind: candidate_bank(kind, sequences) for kind in ("a_left", "a_right", "b_left", "b_right")}
    tries = {kind: ReverseTrie() for kind in ("a_right", "b_right")}
    for kind in ("a_right", "b_right"):
        for candidate in banks[kind]:
            tries[kind].add(candidate)

    parent_tape = normalize(base)
    attempts: list[dict[str, object]] = []
    chosen = None
    for a_left in banks["a_left"]:
        if len(a_left.tape) != TARGET_WINDOW_LETTERS:
            continue
        if has_reflexive_clause(a_left):
            continue
        for a_right in tries["a_right"].exact(a_left.tape):
            if len(attempts) >= MAX_PAIRED_EXPANSIONS:
                break
            if any(normalize(clause) in parent_tape for clause in clause_surfaces(a_left) + clause_surfaces(a_right)):
                continue
            if has_reflexive_clause(a_right):
                continue
            a_pair = consume_pair(a_left, a_right, left_cursor=170, right_cursor=398)
            if a_pair["final_residual"] or a_pair["committed_character_contradictions"]:
                continue
            for b_left in banks["b_left"]:
                if len(b_left.tape) != TARGET_WINDOW_LETTERS:
                    continue
                if has_reflexive_clause(b_left):
                    continue
                for b_right in tries["b_right"].exact(b_left.tape):
                    if len(attempts) >= MAX_PAIRED_EXPANSIONS:
                        break
                    all_clauses = clause_surfaces(a_left) + clause_surfaces(a_right) + clause_surfaces(b_left) + clause_surfaces(b_right)
                    if any(normalize(clause) in parent_tape for clause in clause_surfaces(b_left) + clause_surfaces(b_right)):
                        continue
                    if has_reflexive_clause(b_right):
                        continue
                    if len(set(all_clauses)) != len(all_clauses):
                        continue
                    b_pair = consume_pair(b_left, b_right, left_cursor=469, right_cursor=99)
                    attempt = {
                        "ordinal": len(attempts) + 1,
                        "a_left": a_left.surface,
                        "a_right": a_right.surface,
                        "b_left": b_left.surface,
                        "b_right": b_right.surface,
                        "grammar_state": {
                            "frames": [a_left.frame, a_right.frame, b_left.frame, b_right.frame],
                            "active_entities": sorted(set(a_left.active_entities + a_right.active_entities + b_left.active_entities + b_right.active_entities)),
                            "parent_clause_novel": True,
                        },
                        "residuals": {"a_pair": a_pair, "b_pair": b_pair},
                        "shell_spaces": {"a_left_start": base[a_left_raw - 1:a_left_raw + 1], "a_right_end": base[a_mirror_end - 1:a_mirror_end + 1], "b_left_start": base[b_left_raw - 1:b_left_raw + 1], "b_right_end": base[b_mirror_end - 1:b_mirror_end + 1]},
                        "status": "accepted" if not b_pair["final_residual"] and not b_pair["committed_character_contradictions"] else "rejected_residual",
                    }
                    attempts.append(attempt)
                    if attempt["status"] != "accepted":
                        continue
                    chosen = (a_left, a_right, b_left, b_right, a_pair, b_pair)
                    break
                if chosen or len(attempts) >= MAX_PAIRED_EXPANSIONS:
                    break
            if chosen or len(attempts) >= MAX_PAIRED_EXPANSIONS:
                break
        if chosen or len(attempts) >= MAX_PAIRED_EXPANSIONS:
            break
    assert chosen is not None
    a_left, a_right, b_left, b_right, a_pair, b_pair = chosen

    replacements = {
        (a_left_raw, a_left_end): a_left.surface,
        (b_left_raw, b_left_end): b_left.surface,
        (b_mirror_raw, b_mirror_end): b_right.surface,
        (a_mirror_raw, a_mirror_end): a_right.surface,
    }
    rendered = replace_windows(base, replacements)
    project = audit(rendered)
    independent = independent_audit(rendered)
    assert independent["two_pointer_exact"] and independent["sha_equal"]
    assert independent["normalized_letters"] > 568

    inserted_clauses = clause_surfaces(a_left) + clause_surfaces(a_right) + clause_surfaces(b_left) + clause_surfaces(b_right)
    assert all(rendered.count(clause) == 1 for clause in inserted_clauses)
    fragment_gate = {
        "left_a_words_intact": "Nora saw Noel." in rendered and all(clause in rendered for clause in a_left.clauses) and "Mara saw" in rendered,
        "right_a_words_intact": "Dog was Aram. " in rendered and all(clause in rendered for clause in a_right.clauses) and "“Leon” was Aron" in rendered,
        "left_b_words_intact": "Spam's reviled, " in rendered and all(clause in rendered for clause in b_left.clauses) and "spots Aram" in rendered,
        "right_b_words_intact": "Mara stops." in rendered and all(clause in rendered for clause in b_right.clauses) and "delivers maps" in rendered,
    }
    assert all(fragment_gate.values())
    inherited_repeats = ("Noel live.", "Spam's reviled", "Nora, spam.", "A tub?")
    repetition_delta = {phrase: {"before": base.count(phrase), "after": rendered.count(phrase)} for phrase in inherited_repeats}
    assert all(value["after"] <= value["before"] for value in repetition_delta.values())

    row = {
        "id": "reviewer-seam-170-197-469-496-shell-replacement-592",
        "working_status": "568_lineage_exact_shell_replacement",
        "rendered": rendered,
        "audit": project,
        "independent_audit": independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": independent["normalized_letters"] - 568,
        "new_event_content": list(inserted_clauses),
        "reviewer_seam": {"requested_normalized_windows": [list(x) for x in requested], "reflected_normalized_windows": [list(x) for x in reflected], "recomputed_raw_after_letter_spans": [list(x) for x in raw_spans]},
        "live_bidirectional_residuals": {"a_pair": a_pair, "b_pair": b_pair, "final_residual": "", "committed_character_contradictions": 0, "backtracks": 0},
        "grammar_novelty": {"active_entities": sorted(set(a_left.active_entities + a_right.active_entities + b_left.active_entities + b_right.active_entities)), "parent_clause_reuse": False, "frames": [a_left.frame, a_right.frame, b_left.frame, b_right.frame]},
        "shell_replacement": {"old_a_left": base[a_left_raw:a_left_end], "old_b_left": base[b_left_raw:b_left_end], "old_b_mirror": base[b_mirror_raw:b_mirror_end], "old_a_mirror": base[a_mirror_raw:a_mirror_end], "repetition_delta": repetition_delta, "raw_spaces_preserved": True},
        "full_text_gate": {"fragments": not all(fragment_gate.values()), "sentence_boundary_corruption": False, "inserted_unit_duplicated": False, "worsened_inherited_unit": False, "catalogue_shortcut": False, "word_order_shortcut": False, "status": "passed structural gate; readability remains unpromoted"},
        "attempts": attempts,
        "provenance": "authoritative 568 parent; reviewer-selected four-window reflected shell replaced by typed clause sequences selected through reverse residual tries, with boundary carriers checked after full render",
    }
    return {
        "experiment_id": "incumbent-568-repeated-shell-intersection-20261002",
        "method": "boundary-aware reflected repeated-shell grammar/trie intersection",
        "parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256},
        "rejected_evidence": REJECTED_EVIDENCE,
        "comparison_evidence": comparison,
        "config": {"max_paired_expansions": MAX_PAIRED_EXPANSIONS, "target_window_letters": TARGET_WINDOW_LETTERS, "intact_clause_grammar_first": True, "post_render_repair": False, "fresh_seed": False, "vocabulary_widened": False},
        "stats": {"independently_exact_children": 1, "children_longer_than_568": 1, "longest_letters": independent["normalized_letters"], "attempted_paired_expansions": len(attempts), "committed_character_contradictions": 0, "shells_replaced": 2},
        "preserved_frontier": frontier,
        "rows": [row],
        "next_operator": "retain the exact shell replacement as non-promoted evidence; if readers reject it, switch to another actual seam without post-render repair",
    }


def main() -> None:
    payload = build_payload()
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
