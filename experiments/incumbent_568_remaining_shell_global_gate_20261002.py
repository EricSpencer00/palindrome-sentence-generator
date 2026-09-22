"""Replace the remaining reflected shell with a globally-gated event lattice."""
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
OUT = ROOT / "runs" / "incumbent-568-remaining-shell-global-gate-20261002.json"
PARENT_ID = "outer-causal-scene-568-working-incumbent"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
MAX_EXPANSIONS = 8
REQUESTED_NORMALIZED = ((121, 148), (446, 473))
REQUESTED_RAW = ((156, 193), (614, 652))
SHELL_NORMALIZED = ((108, 135), (433, 460))
SHELL_RAW = ((142, 179), (600, 638))


def validate(entry: dict[str, object]) -> None:
    payload = json.loads((ROOT / str(entry["artifact"])).read_text())
    row = next(row for row in payload["rows"] if row["id"] == entry["id"])
    checked = independent_audit(str(row["rendered"]))
    assert checked["normalized_letters"] == entry["letters"]
    assert checked["two_pointer_exact"] and checked["sha256_forward"] == entry["sha256"]


def sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def frame_counts(text: str) -> Counter[str]:
    return Counter(
        f"{match.group(1).casefold()}|{match.group(2)}|{match.group(3).casefold()}"
        for match in re.finditer(r"\b([A-Z][a-z]+) (sees|stops|spots) ([A-Za-z]+)\.", text)
    )


def raw_clause_counts(text: str) -> dict[str, Counter[str]]:
    """Count ordinary clauses from raw rendered text, before normalization."""
    records = list(re.finditer(r"\b([A-Z][a-z]+) (sees|stops|spots|rewards|maps|delivers) ([A-Za-z]+)\b", text))
    return {
        "subjects": Counter(match.group(1).casefold() for match in records),
        "predicates": Counter(match.group(2) for match in records),
        "objects": Counter(match.group(3).casefold() for match in records),
        "frames": Counter(f"{match.group(1).casefold()}|{match.group(2)}|{match.group(3).casefold()}" for match in records),
    }


def word_counts(text: str, vocabulary: set[str]) -> Counter[str]:
    return Counter(word for word in normalize(text).split() if word in vocabulary)


def global_gate(rendered: str, base: str, introduced: list[Clause], old_left: str, old_right: str, *, max_delta: int = 2) -> dict[str, object]:
    inserted_tapes = [unit.tape for unit in introduced]
    inserted_frames = [unit.frame for unit in introduced]
    introduced_subjects = Counter(unit.subject for unit in introduced)
    introduced_predicates = Counter(unit.predicate for unit in introduced)
    introduced_objects = Counter(unit.object for unit in introduced)
    entity_vocab = set(introduced_subjects)
    predicate_vocab = set(introduced_predicates)
    base_raw_counts, rendered_raw_counts = raw_clause_counts(base), raw_clause_counts(rendered)
    base_entities, rendered_entities = base_raw_counts["subjects"], rendered_raw_counts["subjects"]
    base_objects, rendered_objects = base_raw_counts["objects"], rendered_raw_counts["objects"]
    base_predicates, rendered_predicates = base_raw_counts["predicates"], rendered_raw_counts["predicates"]
    base_frames, rendered_frames = base_raw_counts["frames"], rendered_raw_counts["frames"]
    base_sentences, rendered_sentences = Counter(sentences(base)), Counter(sentences(rendered))
    new_sentences = list((rendered_sentences - base_sentences).elements())
    baseline_lowercase = len(re.findall(r"[.!?]\s+[a-z]", base))
    rendered_lowercase = len(re.findall(r"[.!?]\s+[a-z]", rendered))
    baseline_commas = len(re.findall(r",\s*[A-Z][a-z]+\s+[a-z]+", base))
    rendered_commas = len(re.findall(r",\s*[A-Z][a-z]+\s+[a-z]+", rendered))
    malformed = ("draw no maps", "drawnomaps", "spam onward", "spamonward", "spam on ward")
    clause_pattern = re.compile(r"[A-Z][a-z]+ (?:sees|stops|spots) [A-Za-z]+\.")
    global_entity_deltas = {name: rendered_entities[name] - base_entities[name] for name in sorted(entity_vocab)}
    global_object_deltas = {name: rendered_objects[name] - base_objects[name] for name in sorted(set(introduced_objects))}
    global_predicate_deltas = {name: rendered_predicates[name] - base_predicates[name] for name in sorted(predicate_vocab)}
    global_frame_deltas = {frame: rendered_frames[frame] - base_frames[frame] for frame in sorted(set(inserted_frames))}
    return {
        "lowercase_after_terminal_global": rendered_lowercase <= baseline_lowercase,
        "subjectless_fragment_global": all(clause_pattern.fullmatch(sentence) or sentence in base_sentences for sentence in new_sentences),
        "malformed_phrase_global": not any(token in rendered.casefold() for token in malformed),
        "comma_splice_global": rendered_commas <= baseline_commas,
        "global_subject_count_deltas": global_entity_deltas,
        "global_object_count_deltas": global_object_deltas,
        "global_subject_object_counts_bounded": all(delta <= max_delta for delta in (*global_entity_deltas.values(), *global_object_deltas.values())),
        "global_predicate_count_deltas": global_predicate_deltas,
        "global_predicate_counts_bounded": all(delta <= max_delta for delta in global_predicate_deltas.values()),
        "global_frame_count_deltas": global_frame_deltas,
        "global_frames_novel_and_single": all(base_frames[frame] == 0 and rendered_frames[frame] == 1 for frame in set(inserted_frames)),
        "inherited_repeat_reduction": rendered.count(old_left) < base.count(old_left) and rendered.count(old_right) < base.count(old_right),
        "no_duplicated_inserted_units": all(normalize(rendered).count(tape) == 1 for tape in inserted_tapes),
        "no_repeated_inserted_frames": len(inserted_frames) == len(set(inserted_frames)),
        "semantic_transition_continuity": len({unit.role_family for unit in introduced}) >= 2 and len(set().union(*(set(unit.active_entities) for unit in introduced))) >= 5,
        "raw_clause_counter_source": "rendered_text",
        "count_delta_cap": max_delta,
        "introduced_vs_full_counts": {
            "introduced": {"subjects": dict(introduced_subjects), "predicates": dict(introduced_predicates), "objects": dict(introduced_objects), "frames": dict(Counter(inserted_frames))},
            "parent_full": {name: dict(counter) for name, counter in base_raw_counts.items()},
            "rendered_full": {name: dict(counter) for name, counter in rendered_raw_counts.items()},
            "deltas": {"subjects": global_entity_deltas, "predicates": global_predicate_deltas, "objects": global_object_deltas, "frames": global_frame_deltas},
        },
        "status": "passed",
    }


def build() -> dict[str, object]:
    payload = json.loads(PARENT.read_text())
    parent = next(row for row in payload["rows"] if row["id"] == PARENT_ID)
    base = str(parent["rendered"])
    parent_audit = independent_audit(base)
    assert parent_audit["normalized_letters"] == 568 and parent_audit["sha256_forward"] == PARENT_SHA256
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
        {"artifact": "runs/incumbent-568-repeated-shell-event-lattice-20261002.json", "id": "repeated-shell-event-lattice-594", "letters": 594, "sha256": "9a0bbdb9cdce4beba2397a91dcd5a7900366b7078e60410207580c5d1c3cd22c", "status": "exact_non_reader_promotable", "source_commit": "e3df2e6f", "reason": "preserved clean exact frontier; not reader-ready"},
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

    old_left = "Mara stops rats. A tub? He maps Nora."
    old_right = "Aron, spam. Eh, but a star spots Aram."
    assert base[SHELL_RAW[0][0]:SHELL_RAW[0][1]] == old_left
    assert base[SHELL_RAW[1][0]:SHELL_RAW[1][1]] == old_right
    assert base_tape[SHELL_NORMALIZED[0][0]:SHELL_NORMALIZED[0][1]] == normalize(old_left)
    assert base_tape[SHELL_NORMALIZED[1][0]:SHELL_NORMALIZED[1][1]] == normalize(old_right)
    assert normalize(old_left) == normalize(old_right)[::-1]

    inventory = targeted_inventory()
    trie = ReverseClauseTrie()
    for clause in inventory:
        trie.add(clause)
    # Edge policy varies roles and entities; each right partner is discovered
    # from the reverse trie, never copied from a historical rendering.
    edge_policy = (("observation", "aron", "rats"), ("control", "star", "aidan"), ("control", "nadia", "nora"))
    prior_units = {"nora sees rats.", "nadia stops aron.", "aron stops aidan.", "nadia spots nora.", "nora spots aidan.", "star sees aron."}
    attempts: list[dict[str, object]] = []
    selected: list[tuple[Clause, Clause, dict[str, object]]] = []
    used_frames: set[str] = set()
    used_units: set[str] = set()
    active = {"mara", "rats", "aron", "nora"}
    left_cursor, right_cursor = SHELL_NORMALIZED[0][0], SHELL_NORMALIZED[1][1] - 1
    for role, subject, object_ in edge_policy:
        for left in inventory:
            if (left.role_family, left.subject, left.object) != (role, subject, object_) or left.tape in prior_units:
                continue
            partners = [candidate for candidate in trie.exact(left.tape) if candidate.surface != left.surface]
            right = next((candidate for candidate in partners if candidate.tape not in base_tape and candidate.tape not in prior_units and candidate.frame not in used_frames), None)
            if right is None:
                continue
            residual = consume_pair(left, right, left_cursor, right_cursor)
            next_active = active | set(left.active_entities) | set(right.active_entities)
            gate = {
                "full_clause_novel": left.tape not in base_tape and right.tape not in base_tape and left.tape not in used_units and right.tape not in used_units,
                "full_frame_novel": left.frame not in used_frames and right.frame not in used_frames,
                "predicate_family_varied": not selected or left.role_family != selected[-1][0].role_family or left.predicate != selected[-1][0].predicate,
                "active_entity_transition": bool(next_active - active) or (selected and left.object not in selected[-1][0].active_entities),
                "reverse_residual_empty": not residual["final_residual"] and residual["committed_character_contradictions"] == 0,
            }
            attempts.append({"ordinal": len(attempts) + 1, "left": left.surface, "right_partner": right.surface, "left_grammar_state": left.__dict__, "right_grammar_state": right.__dict__, "cursors": {"left": left_cursor, "right_reverse": right_cursor}, "residual": residual, "gate": gate, "status": "accepted" if all(gate.values()) else "rejected"})
            if not all(gate.values()):
                continue
            selected.append((left, right, residual)); used_frames.update((left.frame, right.frame)); used_units.update((left.tape, right.tape)); active = next_active
            left_cursor += len(left.tape); right_cursor -= len(right.tape)
            break

    assert len(selected) == 3 and len(attempts) <= MAX_EXPANSIONS
    left_units = [left for left, _, _ in selected]
    right_units = [right for _, right, _ in reversed(selected)]
    left_text, right_text = " ".join(unit.surface for unit in left_units), " ".join(unit.surface for unit in right_units)
    rendered = base[:SHELL_RAW[0][0]] + left_text + base[SHELL_RAW[0][1]:SHELL_RAW[1][0]] + right_text + base[SHELL_RAW[1][1]:]
    independent = independent_audit(rendered)
    project = audit(rendered)
    gate = global_gate(rendered, base, left_units + right_units, old_left, old_right)
    assert independent["two_pointer_exact"] and independent["normalized_letters"] > 568
    assert project["project_validator_exact"] and all(value for key, value in gate.items() if key != "status")

    row = {
        "id": "remaining-shell-global-gate-594",
        "working_status": "568_lineage_exact_remaining_shell_clean_child",
        "rendered": rendered,
        "audit": project,
        "independent_audit": independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)), "parent_id": PARENT_ID, "parent_sha256": PARENT_SHA256,
        "growth_over_parent": independent["normalized_letters"] - 568,
        "new_event_content": [unit.surface for unit in left_units + right_units],
        "requested_seam": {"normalized_windows": [list(window) for window in REQUESTED_NORMALIZED], "raw_windows": [list(window) for window in REQUESTED_RAW]},
        "recomputed_seam": {"normalized_shell_spans": [list(window) for window in SHELL_NORMALIZED], "raw_shell_spans": [list(window) for window in SHELL_RAW], "old_left": old_left, "old_right": old_right, "reflection_verified": True},
        "construction": {"reverse_trie": True, "post_render_repair": False, "whole_sentence_search": False, "self_palindrome_seeded": False, "retained_parent_shell": True, "role_families": sorted({unit.role_family for unit in left_units + right_units})},
        "online_state": {"attempted_paired_expansions": len(attempts), "accepted_paired_expansions": len(selected), "attempts": attempts, "final_left_cursor": left_cursor, "final_right_reverse_cursor": right_cursor, "final_residual": "", "active_entities": sorted(active)},
        "full_render_gate": gate,
    }
    return {"experiment_id": "incumbent-568-remaining-shell-global-gate-20261002", "method": "online reverse-trie event lattice with genuinely global full-render gates", "parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256}, "rejected_evidence": rejected, "comparison_evidence": comparison, "config": {"max_paired_expansions": MAX_EXPANSIONS, "post_render_repair": False, "whole_sentence_search": False, "self_palindrome_seeded": False, "historical_608_650_loaded": False}, "stats": {"independently_exact_children": 1, "children_longer_than_568": 1, "longest_letters": independent["normalized_letters"], "attempted_paired_expansions": len(attempts), "accepted_paired_expansions": len(selected)}, "preserved_frontier": frontier, "rows": [row]}


if __name__ == "__main__":
    result = build()
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    row = result["rows"][0]
    print(json.dumps({"artifact": str(OUT.relative_to(ROOT)), "letters": row["independent_audit"]["normalized_letters"], "sha256": row["independent_audit"]["sha256_forward"]}, indent=2))
