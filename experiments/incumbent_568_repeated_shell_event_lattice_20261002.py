"""Construct a clean event lattice in the recorded repeated-shell seam.

This run does not load or copy either historical 608/650 rendering.  It opens
the authoritative 568 tape at the two recorded 27-letter shell spans and
discovers reverse partners through the clause trie.  Each accepted edge is
admitted online only when its residual is empty and its discourse/frequency
gates remain sound.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_568_partial_varied_intersection_20261002 import (
    Clause,
    ReverseClauseTrie,
    consume_pair,
    independent_audit,
    normalize,
    targeted_inventory,
)
from experiments.incumbent_498_deep_clause_transducer_20261002 import audit

PARENT = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT = ROOT / "runs" / "incumbent-568-repeated-shell-event-lattice-20261002.json"
PARENT_ID = "outer-causal-scene-568-working-incumbent"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
MAX_PAIRED_EXPANSIONS = 8
LEFT_WINDOW = (64, 91)
RIGHT_WINDOW = (477, 504)
LEFT_RAW = (83, 120)
RIGHT_RAW = (662, 700)


def validate(entry: dict[str, object]) -> None:
    payload = json.loads((ROOT / str(entry["artifact"])).read_text())
    row = next(row for row in payload["rows"] if row["id"] == entry["id"])
    checked = independent_audit(str(row["rendered"]))
    assert checked["normalized_letters"] == entry["letters"]
    assert checked["two_pointer_exact"] and checked["sha256_forward"] == entry["sha256"]


def full_render_gate(rendered: str, base: str, introduced: list[Clause], old_left: str, old_right: str) -> dict[str, object]:
    normalized = normalize(rendered)
    introduced_tapes = [clause.tape for clause in introduced]
    introduced_frames = [clause.frame for clause in introduced]
    subjects = Counter(clause.subject for clause in introduced)
    predicates = Counter(clause.predicate for clause in introduced)
    objects = Counter(clause.object for clause in introduced)
    blocklist = ("draw no maps", "spam onward", "drawnomaps", "spam on ward")
    sentence_units = [clause.surface for clause in introduced]
    inherited_repeat_worsening = rendered.count(old_left) > base.count(old_left) or rendered.count(old_right) > base.count(old_right)
    return {
        "lowercase_after_terminal": not bool(re.search(r"[.!?]\s+[a-z]", rendered)),
        "subjectless_fragment": not bool(re.search(r"(?:^|[.!?]\s+)(?:and|or|but|maps|ward)\b", rendered, re.I)),
        "comma_splice": not bool(re.search(r",\s*[A-Z][a-z]+\s+[a-z]+", " ".join(sentence_units))),
        "inherited_repeat_worsening": not inherited_repeat_worsening,
        "repeated_full_clauses": len(introduced_tapes) == len(set(introduced_tapes)) and all(normalized.count(tape) == 1 for tape in introduced_tapes),
        "repeated_full_frames": len(introduced_frames) == len(set(introduced_frames)),
        "predicate_frequency_cap": max(predicates.values(), default=0) <= 2,
        "object_frequency_cap": max(objects.values(), default=0) <= 2,
        "subject_frequency_cap": max(subjects.values(), default=0) <= 2,
        "semantic_transition_continuity": len({clause.role_family for clause in introduced}) >= 2 and len(set().union(*(set(clause.active_entities) for clause in introduced))) >= 5,
        "complete_clause_units": all(re.fullmatch(r"[A-Z][a-z]+ (?:sees|stops|spots) [A-Za-z]+\.", unit) for unit in sentence_units),
        "malformed_phrase_blocklist": not any(token in rendered.casefold() for token in blocklist),
        "status": "passed",
    }


def build() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    base = str(parent["rendered"])
    base_audit = independent_audit(base)
    assert base_audit["normalized_letters"] == 568 and base_audit["sha256_forward"] == PARENT_SHA256
    base_tape = normalize(base)

    frontier = [
        {"artifact": "runs/incumbent-560-outer-causal-scene-20261002.json", "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256},
        {"artifact": "runs/incumbent-550-central-event-bridge-20261002.json", "id": "central-distinct-events-560", "letters": 560, "sha256": "b5f98bfb0b44b31d8cbf78727672a74b588980e1fc8f1ff522a2c4ad1d800ccc"},
        {"artifact": "runs/incumbent-550-typed-center-product-20261002.json", "id": "typed-center-25", "letters": 558, "sha256": "29470b5ab408c402e8796530123357fea6a74aa4bdf14f7f1b2a601dbecc94fa"},
        {"artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json", "id": "depth39-longest-f1g1h1r", "letters": 556, "sha256": "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14"},
    ]
    for entry in frontier:
        validate(entry)

    rejected = [
        {"artifact": "runs/incumbent-568-partial-varied-intersection-20261002.json", "id": "partial-110-458-varied-observation-control-594", "letters": 600, "sha256": "9c21bb4f9fd0635b29558a399f5341168d9b34e039c084eaa868724c37a177f7", "status": "exact_prior_evidence", "reason": "preserved prior focused result"},
        {"artifact": "runs/incumbent-568-sentence-boundary-clause-intersection-20261002.json", "id": "sentence-boundary-91-135-clause-growth-588", "letters": 592, "sha256": "e861ab1ea21d1408cf8ac59fc5014e9d858bc72a8ac425fd727687393abdf303", "status": "exact_rejected", "source_commit": "4388c9cd", "reason": "catalogue-like"},
        {"artifact": "runs/incumbent-568-repeated-shell-intersection-20261002.json", "id": "reviewer-seam-170-197-469-496-shell-replacement-592", "letters": 592, "sha256": "4c06d5151ffef85d6ebfa1e1199e0594ac8be00d4cd3b61e20e387020d55e723", "status": "exact_rejected", "reason": "structural gate defect"},
        {"artifact": "runs/incumbent-568-token-boundary-intersection-20261002.json", "id": "token-boundary-64-nadia-stops-rats-then-star-spots-aidan", "letters": 596, "sha256": "514f284bbcc88c5eda7b99dd235ca4f5845c45825b943652008e6d6ed6a97313", "status": "exact_rejected", "reason": "not reader-promotable"},
        {"artifact": "runs/incumbent-568-obligation-indexed-intersection-20261002.json", "id": "seam-100-nadia-stops-rats-then-star-spots-aidan", "letters": 596, "sha256": "d590e5cb08854dd2795ba65ee46ef0292764a2fcf469a8f9ced349f48745e1dc", "status": "rejected", "reason": "split-word fragments"},
    ]
    for entry in rejected:
        validate(entry)
    comparison = {"artifact": "runs/incumbent-666-linked-scene-lattice-20260922.json", "id": "bidirectional-typed-trie-alternative-666", "letters": 666, "sha256": "bab693719482af36c7e223a687f94552ad3efda6825d481014134a7d7ae7148d", "source_commit": "9cb68296"}
    validate(comparison)

    old_left = "Mara stops rats. A tub? He maps Aron."
    old_right = "Nora, spam. Eh, but a star spots Aram."
    assert base[LEFT_RAW[0]:LEFT_RAW[1]] == old_left
    assert base[RIGHT_RAW[0]:RIGHT_RAW[1]] == old_right
    assert normalize(old_left) == normalize(old_right)[::-1]
    assert base_tape[LEFT_WINDOW[0]:LEFT_WINDOW[1]] == normalize(old_left)
    assert base_tape[RIGHT_WINDOW[0]:RIGHT_WINDOW[1]] == normalize(old_right)

    inventory = targeted_inventory()
    trie = ReverseClauseTrie()
    for clause in inventory:
        trie.add(clause)
    # The lattice is selected by discourse edges; partners remain trie-derived.
    edge_policy = (
        ("observation", "nora", "rats"),
        ("control", "nadia", "aron"),
        ("control", "aron", "aidan"),
    )
    attempts: list[dict[str, object]] = []
    selected: list[tuple[Clause, Clause, dict[str, object]]] = []
    used_frames: set[str] = set()
    used_clauses: set[str] = set()
    active = {"mara", "rats", "aron", "nora"}
    subjects: Counter[str] = Counter()
    predicates: Counter[str] = Counter()
    objects: Counter[str] = Counter()
    left_cursor, right_cursor = LEFT_WINDOW[0], RIGHT_WINDOW[1] - 1
    for role, subject, object_ in edge_policy:
        for left in inventory:
            if left.role_family != role or left.subject != subject or left.object != object_:
                continue
            partners = [candidate for candidate in trie.exact(left.tape) if candidate.surface != left.surface]
            right = next((candidate for candidate in partners if candidate.frame not in used_frames and candidate.surface not in used_clauses), None)
            if right is None:
                continue
            pair = (left, right)
            next_subjects = subjects + Counter((left.subject, right.subject))
            next_predicates = predicates + Counter((left.predicate, right.predicate))
            next_objects = objects + Counter((left.object, right.object))
            residual = consume_pair(left, right, left_cursor, right_cursor)
            gate = {
                "full_clause_novel": left.tape not in base_tape and right.tape not in base_tape and left.surface not in used_clauses and right.surface not in used_clauses,
                "full_frame_novel": left.frame not in used_frames and right.frame not in used_frames,
                "predicate_frequency_cap": max(next_predicates.values(), default=0) <= 2,
                "object_frequency_cap": max(next_objects.values(), default=0) <= 2,
                "subject_frequency_cap": max(next_subjects.values(), default=0) <= 2,
                "semantic_transition": bool(set(left.active_entities + right.active_entities) - active) or role != (selected[-1][0].role_family if selected else None) or (selected and left.object not in selected[-1][0].active_entities),
                "reverse_obligation_exact": not residual["final_residual"] and residual["committed_character_contradictions"] == 0,
            }
            attempt = {"ordinal": len(attempts) + 1, "left": left.surface, "right_partner": right.surface, "cursors": {"left": left_cursor, "right_reverse": right_cursor}, "residual": residual, "active_before": sorted(active), "gate": gate, "status": "accepted" if all(gate.values()) else "rejected_gate"}
            attempts.append(attempt)
            if attempt["status"] != "accepted":
                continue
            selected.append((left, right, residual))
            used_clauses.update((left.surface, right.surface)); used_frames.update((left.frame, right.frame))
            subjects.update((left.subject, right.subject)); predicates.update((left.predicate, right.predicate)); objects.update((left.object, right.object))
            active.update(left.active_entities); active.update(right.active_entities)
            left_cursor += len(left.tape); right_cursor -= len(right.tape)
            break

    assert len(selected) == 3 and len(attempts) <= MAX_PAIRED_EXPANSIONS
    left_units = [left for left, _, _ in selected]
    right_units = [right for _, right, _ in reversed(selected)]
    left_text, right_text = " ".join(unit.surface for unit in left_units), " ".join(unit.surface for unit in right_units)
    rendered = base[:LEFT_RAW[0]] + left_text + base[LEFT_RAW[1]:RIGHT_RAW[0]] + right_text + base[RIGHT_RAW[1]:]
    independent = independent_audit(rendered)
    project = audit(rendered)
    full_gate = full_render_gate(rendered, base, left_units + right_units, old_left, old_right)
    assert independent["two_pointer_exact"] and independent["normalized_letters"] > 568
    assert project["project_validator_exact"] and all(value for key, value in full_gate.items() if key != "status")

    row = {
        "id": "repeated-shell-event-lattice-594",
        "working_status": "568_lineage_exact_clean_event_lattice",
        "rendered": rendered,
        "audit": project,
        "independent_audit": independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)), "parent_id": PARENT_ID, "parent_sha256": PARENT_SHA256,
        "growth_over_parent": independent["normalized_letters"] - 568,
        "new_event_content": [unit.surface for unit in left_units + right_units],
        "seam": {"normalized_windows": [list(LEFT_WINDOW), list(RIGHT_WINDOW)], "raw_spans": [list(LEFT_RAW), list(RIGHT_RAW)], "old_left": old_left, "old_right": old_right, "reflection_verified": True},
        "lattice": {"role_families": sorted({unit.role_family for unit in left_units + right_units}), "active_entities": sorted(active), "reverse_trie": True, "edge_policy": [list(edge) for edge in edge_policy]},
        "online_state": {"attempted_paired_expansions": len(attempts), "accepted_paired_expansions": len(selected), "attempts": attempts, "final_left_cursor": left_cursor, "final_right_reverse_cursor": right_cursor, "final_residual": "", "subject_frequency": dict(subjects), "predicate_frequency": dict(predicates), "object_frequency": dict(objects)},
        "full_render_gate": full_gate,
        "malformed_exact_rejection_policy": {"blocked_phrases": ["Draw no maps", "Spam onward"], "applied_before_admission": True, "historical_608_650_loaded": False},
    }
    return {"experiment_id": "incumbent-568-repeated-shell-event-lattice-20261002", "method": "targeted discourse-connected event lattice with online reverse obligations in the recorded repeated shell", "parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256}, "rejected_evidence": rejected, "comparison_evidence": comparison, "config": {"max_paired_expansions": MAX_PAIRED_EXPANSIONS, "post_render_repair": False, "fresh_seed": False, "vocabulary_widened": False, "historical_608_650_loaded": False}, "stats": {"independently_exact_children": 1, "children_longer_than_568": 1, "longest_letters": independent["normalized_letters"], "attempted_paired_expansions": len(attempts), "accepted_paired_expansions": len(selected)}, "preserved_frontier": frontier, "rows": [row]}


if __name__ == "__main__":
    result = build()
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    row = result["rows"][0]
    print(json.dumps({"artifact": str(OUT.relative_to(ROOT)), "letters": row["independent_audit"]["normalized_letters"], "sha256": row["independent_audit"]["sha256_forward"]}, indent=2))
