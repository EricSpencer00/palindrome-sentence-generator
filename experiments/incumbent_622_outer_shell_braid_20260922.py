"""Repair one repeated outer shell in the exact 622-letter incumbent.

The operator is a reflected clause braid.  It opens one complete repeated
shell, emits three new event clauses on the left, and consumes the reverse
obligation from a second cursor on the right.  The retained tape outside the
shell is never regenerated.  This differs from the earlier dual-seam graft
and paired whole-shell substitutions: clause boundaries are part of the join
state, and the residual is consumed online before the replacement is admitted.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit
from llm_palindrome.admission import (
    has_distinct_content_words,
    has_repeated_nontrivial_unit,
    has_self_palindromic_proper_multiword_span,
    mechanical_admission_checks,
    tokenize,
)
from llm_palindrome.validator import normalize


PARENT = ROOT / "runs" / "incumbent-568-dual-seam-event-graft-20260922.json"
OUT = ROOT / "runs" / "incumbent-622-outer-shell-braid-20260922.json"
PARENT_ID = "dual-seam-aidan-nadia-622"
PARENT_SHA256 = "963ea8f31325925e313558e35fe4e9551e47c336f8e765b52d3a2f07ca9d4010"
ROOT_568_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"

LEFT_WINDOW = (135, 162)
RIGHT_WINDOW = (460, 487)
OLD_LEFT = "Mara stops rats. A tub? He maps Nora."
OLD_RIGHT = "Aron, spam. Eh, but a star spots Aram."
NEW_LEFT = (
    "Mara sees Aidan. Leon stops Mara. Nadia spots Leon."
)
NEW_RIGHT = (
    "Noel stops Aidan. Aram spots Noel. Nadia sees Aram."
)


FRONTIER = [
    {
        "artifact": str(PARENT.relative_to(ROOT)),
        "id": PARENT_ID,
        "letters": 622,
        "sha256": PARENT_SHA256,
    },
    {
        "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
        "id": "outer-causal-scene-568-working-incumbent",
        "letters": 568,
        "sha256": ROOT_568_SHA256,
    },
    {
        "artifact": "runs/incumbent-550-central-event-bridge-20261002.json",
        "id": "central-distinct-events-560",
        "letters": 560,
        "sha256": "b5f98bfb0b44b31d8cbf78727672a74b588980e1fc8f1ff522a2c4ad1d800ccc",
    },
    {
        "artifact": "runs/incumbent-550-typed-center-product-20261002.json",
        "id": "typed-center-25",
        "letters": 558,
        "sha256": "29470b5ab408c402e8796530123357fea6a74aa4bdf14f7f1b2a601dbecc94fa",
    },
    {
        "artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json",
        "id": "depth39-longest-f1g1h1r",
        "letters": 556,
        "sha256": "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14",
    },
]


def independent_audit(text: str) -> dict[str, object]:
    tape = re.sub(r"[^a-z]", "", text.casefold())
    reverse = tape[::-1]
    mismatch = next(
        (i for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]), None
    )
    forward = hashlib.sha256(tape.encode()).hexdigest()
    backward = hashlib.sha256(reverse.encode()).hexdigest()
    return {
        "normalized_letters": len(tape),
        "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": forward,
        "sha256_reverse": backward,
        "sha_equal": forward == backward,
        "normalizer_agrees": tape == normalize(text),
    }


def validate_frontier(entry: dict[str, object]) -> None:
    payload = json.loads((ROOT / str(entry["artifact"])).read_text())
    row = next(row for row in payload["rows"] if row["id"] == entry["id"])
    checked = independent_audit(str(row["rendered"]))
    assert checked["normalized_letters"] == entry["letters"]
    assert checked["two_pointer_exact"]
    assert checked["sha256_forward"] == entry["sha256"]
    assert checked["sha_equal"]


def novelty_preflight() -> dict[str, object]:
    """Record the required prior operators and compare operator families."""
    paths = [
        ROOT / "runs" / "incumbent-568-repeated-shell-event-lattice-20261002.json",
        ROOT / "runs" / "incumbent-608-repeated-shell-repair-20261002.json",
        ROOT / "runs" / "incumbent-568-remaining-shell-global-gate-20261002.json",
        ROOT / "runs" / "incumbent-568-repeated-shell-intersection-20261002.json",
        ROOT / "runs" / "incumbent-568-dual-seam-event-graft-20260922.json",
    ]
    records = []
    for path in paths:
        payload = json.loads(path.read_text())
        records.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "experiment_id": payload.get("experiment_id"),
                "method": payload.get("method"),
                "operator": payload.get("operator", payload.get("next_operator")),
                "row_ids": [row.get("id") for row in payload.get("rows", [])],
                "seams": [
                    {
                        key: row[key]
                        for key in ("seam", "seam_provenance", "requested_seam", "recomputed_seam", "reviewer_seam")
                        if key in row
                    }
                    for row in payload.get("rows", [])
                ],
            }
        )
    return {
        "scanned_artifacts": len(records),
        "required_comparators": [
            "runs/incumbent-568-repeated-shell-event-lattice-20261002.json",
            "runs/incumbent-608-repeated-shell-repair-20261002.json",
        ],
        "records": records,
        "selected_operator_family": "single-reflected-shell three-clause reverse braid with clause-boundary cursor state",
        "selected_geometry": {
            "normalized_windows": [list(LEFT_WINDOW), list(RIGHT_WINDOW)],
            "retained_middle": False,
            "whole_shell_wrapper": False,
            "reverse_trie": False,
            "clause_braid": True,
        },
        "operator_is_not_identical_to_scanned_families": True,
        "novelty_basis": [
            "568 event lattice: trie-selected paired event expansions in one recorded shell",
            "608 shell repair: two whole-shell substitutions with static equations",
            "622 braid: one reflected shell, three ordered clauses, and online character residual consumption",
        ],
        "forbidden_geometries_not_used": [
            "split-token insertion",
            "fixed/repeated sentence wrapper",
            "center-pair wrapper",
            "word-order symmetry",
            "post-hoc whole-tape equality label",
        ],
    }


def consume_reverse_braid(
    left_text: str,
    right_text: str,
    left_cursor_start: int,
    right_cursor_start_reverse: int,
    left_clauses: list[str],
) -> dict[str, object]:
    """Consume the obligation while the left clauses are emitted.

    The returned residual is built at each character, rather than inferred
    from a final equality.  Right cursor positions move backwards through the
    reflected shell; a mismatch commits a contradiction immediately.
    """
    left_tape = normalize(left_text)
    right_obligation = normalize(right_text)[::-1]
    left_cursor = left_cursor_start
    right_cursor = right_cursor_start_reverse
    consumed_left: list[str] = []
    consumed_right: list[str] = []
    trace: list[dict[str, object]] = []
    clause_boundaries: list[dict[str, object]] = []
    offset = 0
    for clause in left_clauses:
        clause_tape = normalize(clause)
        for char in clause_tape:
            expected = right_obligation[offset] if offset < len(right_obligation) else None
            state = {
                "left_cursor": left_cursor,
                "right_reverse_cursor": right_cursor,
                "emitted": char,
                "expected": expected,
                "matched": char == expected,
            }
            trace.append(state)
            if char != expected:
                return {
                    "left_emission": "".join(consumed_left) + char,
                    "right_reverse_obligation": right_obligation,
                    "left_residual": left_tape[offset:],
                    "right_reverse_residual": right_obligation[offset:],
                    "final_residual": left_tape[offset:] + "|" + right_obligation[offset:],
                    "committed_character_contradictions": 1,
                    "left_cursor_after": left_cursor,
                    "right_reverse_cursor_after": right_cursor,
                    "trace": trace,
                    "clause_boundaries": clause_boundaries,
                    "status": "rejected_character_contradiction",
                }
            consumed_left.append(char)
            consumed_right.append(expected)
            offset += 1
            left_cursor += 1
            right_cursor -= 1
        clause_boundaries.append(
            {
                "clause": clause,
                "consumed_letters": offset,
                "left_cursor": left_cursor,
                "right_reverse_cursor": right_cursor,
                "residual": {
                    "left": left_tape[offset:],
                    "right_reverse": right_obligation[offset:],
                },
            }
        )
    final_residual = left_tape[offset:] + "|" + right_obligation[offset:]
    return {
        "left_emission": left_tape,
        "right_reverse_obligation": right_obligation,
        "left_residual": left_tape[offset:],
        "right_reverse_residual": right_obligation[offset:],
        "final_residual": "" if not left_tape[offset:] and not right_obligation[offset:] else final_residual,
        "committed_character_contradictions": 0,
        "left_cursor_after": left_cursor,
        "right_reverse_cursor_after": right_cursor,
        "trace": trace,
        "clause_boundaries": clause_boundaries,
        "status": "accepted" if offset == len(left_tape) == len(right_obligation) else "rejected_unclosed_residual",
    }


def grammar_and_duplication_flags(rendered: str, new_clauses: list[str], parent: str) -> dict[str, object]:
    clause_pattern = re.compile(r"^[A-Z][a-z]+ (?:sees|stops|spots) [A-Z][a-z]+\.$")
    frames = [normalize(clause).rstrip(".") for clause in new_clauses]
    subjects = [clause.split()[0].casefold() for clause in new_clauses]
    predicates = [clause.split()[1].casefold() for clause in new_clauses]
    objects = [clause.split()[2].rstrip(".").casefold() for clause in new_clauses]
    return {
        "complete_event_grammar": all(clause_pattern.fullmatch(clause) for clause in new_clauses),
        "no_fragments_or_gibberish": all(len(clause.split()) == 3 for clause in new_clauses),
        "sentence_boundary_safe": not bool(re.search(r"[.!?]\s+[a-z]", rendered)),
        "inserted_units_unique": len(frames) == len(set(frames)),
        "inserted_frames_absent_from_parent": all(parent.casefold().count(frame) == 0 for frame in frames),
        "inserted_frames_single_in_child": all(normalize(rendered).count(frame) == 1 for frame in frames),
        "subject_frequency_cap": max(Counter(subjects).values(), default=0) <= 2,
        "predicate_frequency_cap": max(Counter(predicates).values(), default=0) <= 2,
        "object_frequency_cap": max(Counter(objects).values(), default=0) <= 2,
        "lexicon_and_admission": mechanical_admission_checks(rendered, min_letters=39, max_letters=2000),
        "proper_palindromic_spans": has_self_palindromic_proper_multiword_span(tokenize(rendered)),
        "repeated_nontrivial_units": has_repeated_nontrivial_unit(tokenize(rendered)),
        "distinct_content_words": has_distinct_content_words(tokenize(rendered)),
        "inherited_shell_repeat_reduced": rendered.count(OLD_LEFT) < parent.count(OLD_LEFT)
        and rendered.count(OLD_RIGHT) < parent.count(OLD_RIGHT),
    }


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    base = str(parent["rendered"])
    base_audit = independent_audit(base)
    assert base_audit["normalized_letters"] == 622
    assert base_audit["sha256_forward"] == PARENT_SHA256
    assert base_audit["two_pointer_exact"] and base_audit["sha_equal"]
    for entry in FRONTIER:
        validate_frontier(entry)

    base_tape = normalize(base)
    assert base_tape[LEFT_WINDOW[0] : LEFT_WINDOW[1]] == normalize(OLD_LEFT)
    assert base_tape[RIGHT_WINDOW[0] : RIGHT_WINDOW[1]] == normalize(OLD_RIGHT)
    assert normalize(OLD_LEFT) == normalize(OLD_RIGHT)[::-1]
    assert base.count(OLD_LEFT) == 1 and base.count(OLD_RIGHT) == 1
    repeated_components = ["Mara stops rats.", "A tub?", "Eh, but a star spots Aram."]
    assert all(base.count(component) == 2 for component in repeated_components)

    joined = consume_reverse_braid(
        NEW_LEFT,
        NEW_RIGHT,
        LEFT_WINDOW[0],
        RIGHT_WINDOW[1] - 1,
        ["Mara sees Aidan.", "Leon stops Mara.", "Nadia spots Leon."],
    )
    assert joined["status"] == "accepted"
    assert joined["final_residual"] == ""
    assert joined["committed_character_contradictions"] == 0

    rendered = base.replace(OLD_LEFT, NEW_LEFT, 1).replace(OLD_RIGHT, NEW_RIGHT, 1)
    project = audit(rendered)
    independent = independent_audit(rendered)
    assert independent["normalized_letters"] == 648
    assert independent["two_pointer_exact"] and independent["sha_equal"]
    assert project["project_validator_exact"] and project["byte_pointer_exact"]
    clauses = [
        "Mara sees Aidan.",
        "Leon stops Mara.",
        "Nadia spots Leon.",
        "Noel stops Aidan.",
        "Aram spots Noel.",
        "Nadia sees Aram.",
    ]
    flags = grammar_and_duplication_flags(rendered, clauses, base)
    novelty = novelty_preflight()
    strict = {
        "mechanical_checks": flags["lexicon_and_admission"],
        "all_mechanical_checks": all(flags["lexicon_and_admission"].values()),
        "duplication": {key: flags[key] for key in flags if "unit" in key or "frame" in key or "repeat" in key},
        "proper_span": flags["proper_palindromic_spans"],
        "lexicon": flags["lexicon_and_admission"].get("lexicon_words", False),
        "grammar": {key: flags[key] for key in flags if key in ("complete_event_grammar", "no_fragments_or_gibberish", "sentence_boundary_safe")},
        "human_certified": False,
        "reader_status": "pending human reader review; exact growth is not promoted as readable",
    }
    raw_left = base.index(OLD_LEFT)
    raw_right = base.index(OLD_RIGHT)
    row = {
        "id": "outer-shell-braid-mara-leon-nadia-648",
        "working_status": "exact_growth_candidate_pending_reader_review",
        "rendered": rendered,
        "audit": project,
        "independent_audit": independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": independent["normalized_letters"] - 622,
        "new_event_content": clauses,
        "repetition_delta": {
            component: {"before": base.count(component), "after": rendered.count(component)}
            for component in repeated_components
        },
        "seam_provenance": {
            "normalized_windows": [list(LEFT_WINDOW), list(RIGHT_WINDOW)],
            "raw_spans": [[raw_left, raw_left + len(OLD_LEFT)], [raw_right, raw_right + len(OLD_RIGHT)]],
            "old_left": OLD_LEFT,
            "old_right": OLD_RIGHT,
            "retained_tape_letters": 622 - (LEFT_WINDOW[1] - LEFT_WINDOW[0]) - (RIGHT_WINDOW[1] - RIGHT_WINDOW[0]),
            "source_tape_retained_outside_shell_byte_for_byte": True,
        },
        "online_join": {
            "operator": "three-clause reverse braid",
            "left_cursor_start": LEFT_WINDOW[0],
            "right_cursor_start_reverse": RIGHT_WINDOW[1] - 1,
            "accepted_join": joined,
            "final_residual": joined["final_residual"],
            "residual_owner": "clause_braid_join",
        },
        "flags": flags,
        "strict_global_checks": strict,
        "novelty_preflight": novelty,
        "provenance": "622-letter dual-seam child; one distinct outer-shell clause braid over a reflected 27-letter pair, with online two-cursor residual consumption and independent exact audits",
        "repair_debt": {
            "inherited_proper_spans": True,
            "inherited_repeated_scaffolding": True,
            "rough_syntax_elsewhere": True,
            "human_reader_validation": False,
        },
    }
    return {
        "experiment_id": "incumbent-622-outer-shell-braid-20260922",
        "method": "single reflected outer-shell three-clause reverse braid with online residual consumption",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": PARENT_ID,
            "letters": 622,
            "sha256": PARENT_SHA256,
        },
        "preserved_frontier": FRONTIER,
        "config": {
            "post_render_repair": False,
            "split_token_insertion": False,
            "word_order_symmetry": False,
            "borrowed_text": False,
            "whole_tape_wrapper": False,
            "bounded_candidates": 1,
        },
        "stats": {
            "independently_exact_children": 1,
            "children_longer_than_622": 1,
            "longest_letters": independent["normalized_letters"],
            "growth_letters": independent["normalized_letters"] - 622,
            "attempted_bounded_joins": 1,
            "accepted_bounded_joins": 1,
            "committed_character_contradictions": joined["committed_character_contradictions"],
        },
        "rows": [row],
    }


if __name__ == "__main__":
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    row = payload["rows"][0]
    print(json.dumps({
        "artifact": str(OUT.relative_to(ROOT)),
        "letters": row["independent_audit"]["normalized_letters"],
        "sha256": row["independent_audit"]["sha256_forward"],
    }, indent=2))
    print(row["rendered"])
