"""Record a bounded, authored dual-sentence graft equation on the 568 tape.

This is not an online cursor search: the complete reciprocal event sequences
are authored first, compared as whole strings, then rendered around a retained
middle.  The artifact exists as exact-growth evidence and a negative method
record, not as a novel residual-owning construction operator.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
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
from llm_palindrome.validator import is_palindrome, normalize


PARENT = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT = ROOT / "runs" / "incumbent-568-dual-seam-event-graft-20260922.json"
PARENT_ID = "outer-causal-scene-568-working-incumbent"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
LEFT_CUT = 108
RIGHT_CUT = 460

FRONTIER = [
    {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256},
    {"artifact": "runs/incumbent-550-central-event-bridge-20261002.json", "id": "central-distinct-events-560", "letters": 560, "sha256": "b5f98bfb0b44b31d8cbf78727672a74b588980e1fc8f1ff522a2c4ad1d800ccc"},
    {"artifact": "runs/incumbent-550-typed-center-product-20261002.json", "id": "typed-center-25", "letters": 558, "sha256": "29470b5ab408c402e8796530123357fea6a74aa4bdf14f7f1b2a601dbecc94fa"},
    {"artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json", "id": "depth39-longest-f1g1h1r", "letters": 556, "sha256": "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14"},
]


def independent_audit(text: str) -> dict[str, object]:
    tape = re.sub(r"[^a-z]", "", text.casefold())
    reverse = tape[::-1]
    mismatch = next((i for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]), None)
    return {
        "normalized_letters": len(tape),
        "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(reverse.encode()).hexdigest(),
        "sha_equal": hashlib.sha256(tape.encode()).hexdigest() == hashlib.sha256(reverse.encode()).hexdigest(),
        "normalizer_agrees": tape == normalize(text),
    }


def raw_after_letters(text: str, count: int) -> int:
    seen = 0
    for index, char in enumerate(text):
        if char.isascii() and char.isalpha():
            seen += 1
            if seen == count:
                boundary = index + 1
                while boundary < len(text) and not text[boundary].isalpha():
                    boundary += 1
                return boundary
    raise ValueError(count)


def validate_frontier(entry: dict[str, object]) -> None:
    payload = json.loads((ROOT / str(entry["artifact"])).read_text())
    row = next(row for row in payload["rows"] if row["id"] == entry["id"])
    checked = independent_audit(str(row["rendered"]))
    assert checked["normalized_letters"] == entry["letters"]
    assert checked["two_pointer_exact"] and checked["sha256_forward"] == entry["sha256"]
    assert checked["sha_equal"]


def novelty_preflight(candidate: dict[str, object]) -> dict[str, object]:
    """Scan the local 568/608 lineage and report overlap instead of claiming novelty."""
    paths = sorted(set(
        list((ROOT / "runs").glob("incumbent-568-*.json"))
        + list((ROOT / "runs").glob("incumbent-608-*.json"))
    ))
    target_events = candidate["left_events"] + candidate["right_events"]
    normalized_events = {event: normalize(str(event)) for event in target_events}
    records = []
    event_hits: dict[str, list[dict[str, str]]] = {event: [] for event in target_events}
    for path in paths:
        if path == OUT:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        entry = {
            "artifact": str(path.relative_to(ROOT)),
            "experiment_id": payload.get("experiment_id"),
            "method": payload.get("method"),
            "row_ids": [],
            "same_normalized_cuts": [],
        }
        for row in payload.get("rows", []):
            row_id = str(row.get("id", "unknown"))
            entry["row_ids"].append(row_id)
            rendered = str(row.get("rendered", ""))
            normalized_rendered = normalize(rendered)
            for event, normalized_event in normalized_events.items():
                if normalized_event and normalized_event in normalized_rendered:
                    event_hits[event].append({"artifact": entry["artifact"], "row_id": row_id})
            # Look for explicit normalized cut declarations anywhere under the row.
            def visit(value: object) -> None:
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key in {"normalized_cuts", "selected_normalized_cuts"} and child == [LEFT_CUT, RIGHT_CUT]:
                            entry["same_normalized_cuts"].append(row_id)
                        visit(child)
                elif isinstance(value, list):
                    for child in value:
                        visit(child)
            visit(row)
        records.append(entry)
    exact_event_hits = {event: hits for event, hits in event_hits.items() if hits}
    same_cut_records = [record["artifact"] for record in records if record["same_normalized_cuts"]]
    # Manual evidence from the independent review: this seam family and two of
    # these clauses had already appeared in earlier shell/clause operators.
    known_same_geometry = [
        "runs/incumbent-568-remaining-shell-global-gate-20261002.json",
        "runs/incumbent-568-repeated-shell-event-lattice-20261002.json",
    ]
    known_same_geometry = [path for path in known_same_geometry if (ROOT / path).is_file()]
    return {
        "scanned_artifacts": len(records),
        "scope_note": "Scanned every local incumbent-568-* and incumbent-608-* JSON artifact; prior candidate tapes and explicit cut metadata were checked. This does not establish novelty beyond the saved local lineage.",
        "records": records,
        "selected_geometry": "authored complete reciprocal event sequences around a retained middle; this is a bounded graft instance, not a new search-operator family",
        "selected_normalized_cuts": [LEFT_CUT, RIGHT_CUT],
        "same_cut_artifacts": sorted(set(same_cut_records + known_same_geometry)),
        "prior_exact_event_hits": exact_event_hits,
        "event_phrases_absent_from_prior_lineage": [event for event in target_events if event not in exact_event_hits],
        "novelty_status": "not novel as an operator family; event clauses overlap prior saved candidates",
        "forbidden_geometries_not_used": [
            "split-token insertion", "fixed/repeated sentence-shell replacement", "center-pair wrapper", "phrase-bank sweep"
        ],
    }


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    base = str(parent["rendered"])
    base_audit = independent_audit(base)
    assert base_audit["normalized_letters"] == 568
    assert base_audit["sha256_forward"] == PARENT_SHA256 and base_audit["two_pointer_exact"]
    for entry in FRONTIER:
        validate_frontier(entry)

    left_raw = raw_after_letters(base, LEFT_CUT)
    right_raw = raw_after_letters(base, RIGHT_CUT)
    assert base[:left_raw].rstrip().endswith("Aidan delivers maps.")
    assert base[right_raw:].startswith("Spam's reviled, Nadia.")
    assert LEFT_CUT + RIGHT_CUT == 568
    retained = base[left_raw:right_raw]
    assert normalize(base[:left_raw]) == normalize(base[right_raw:])[::-1]

    candidates = [
        {
            "id": "dual-seam-aidan-nadia-622",
            "left_events": ["Aidan sees Mara.", "Nadia stops Aron."],
            "right_events": ["Nora spots Aidan.", "Aram sees Nadia."],
        },
        {
            "id": "dual-seam-leon-nadia-622",
            "left_events": ["Leon sees Mara.", "Nadia stops Aron."],
            "right_events": ["Nora spots Aidan.", "Aram sees Noel."],
        },
        {
            "id": "dual-seam-aidan-nadia-sees-622",
            "left_events": ["Aidan sees Mara.", "Nadia sees Leon."],
            "right_events": ["Noel sees Aidan.", "Aram sees Nadia."],
        },
    ]
    attempts = []
    selected = None
    for candidate in candidates:
        left = " ".join(candidate["left_events"])
        right = " ".join(candidate["right_events"])
        left_tape, right_tape = normalize(left), normalize(right)
        authored_roles = {
            "left_subjects": [event.split()[0] for event in candidate["left_events"]],
            "left_predicates": [event.split()[1] for event in candidate["left_events"]],
            "left_objects": [event.split()[-1].rstrip(".") for event in candidate["left_events"]],
            "right_subjects": [event.split()[0] for event in candidate["right_events"]],
            "right_predicates": [event.split()[1] for event in candidate["right_events"]],
            "right_objects": [event.split()[-1].rstrip(".") for event in candidate["right_events"]],
        }
        exact_equation = left_tape == right_tape[::-1]
        attempts.append({
            "id": candidate["id"], "left_events": candidate["left_events"], "right_events": candidate["right_events"],
            "proposed_cut_offsets": {"left_start": LEFT_CUT, "right_reverse_start": RIGHT_CUT - 1, "authored_left_end": LEFT_CUT + len(left_tape), "authored_right_reverse_end": RIGHT_CUT - len(right_tape)},
            "full_equation_left_side": left_tape, "full_equation_right_side": right_tape[::-1],
            "equation_checked_after_authoring": True, "exact_equation": exact_equation,
            "authored_event_roles": authored_roles,
            "status": "accepted" if exact_equation and selected is None else ("exact_not_selected" if exact_equation else "rejected_equation"),
        })
        if exact_equation and selected is None:
            selected = (candidate, left, right, left_tape, right_tape)
    assert selected is not None
    candidate, left, right, left_tape, right_tape = selected
    rendered = (
        base[:left_raw].rstrip() + " " + left + " " + retained.strip()
        + " " + right + " " + base[right_raw:].lstrip()
    )
    project = audit(rendered)
    independent = independent_audit(rendered)
    assert independent["normalized_letters"] > 568 and independent["two_pointer_exact"]
    assert project["project_validator_exact"] and project["byte_pointer_exact"]
    assert independent["sha_equal"]
    units = tokenize(rendered)
    mechanical = mechanical_admission_checks(rendered, min_letters=39, max_letters=2000)
    strict = {
        "mechanical_checks": mechanical,
        "all_mechanical_checks": all(mechanical.values()),
        "proper_palindromic_spans": has_self_palindromic_proper_multiword_span(units),
        "repeated_nontrivial_units": has_repeated_nontrivial_unit(units),
        "distinct_content_words": has_distinct_content_words(units),
        "human_certified": False,
        "status": "exact growth evidence; inherited shortcut/readability debt is recorded, not silently cleared",
    }
    novelty = novelty_preflight(candidate)
    return {
        "experiment_id": "incumbent-568-dual-seam-event-graft-20260922",
        "method": "bounded authored reciprocal event-graft equation with retained middle tape (not an online residual search)",
        "novelty_preflight": novelty,
        "parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256},
        "preserved_frontier": FRONTIER,
        "config": {"post_render_repair": False, "split_token_insertion": False, "center_wrapper": False, "phrase_bank_sweep": False, "bounded_candidates": len(candidates)},
        "stats": {"independently_exact_children": 1, "children_longer_than_568": 1, "longest_letters": independent["normalized_letters"], "attempted_candidates": len(attempts), "accepted_candidates": 1},
        "rows": [{
            "id": candidate["id"], "rendered": rendered, "audit": project, "independent_audit": independent,
            "parent_artifact": str(PARENT.relative_to(ROOT)), "parent_sha256": PARENT_SHA256,
            "growth_over_parent": independent["normalized_letters"] - 568,
            "new_event_content": candidate["left_events"] + candidate["right_events"],
            "seam_provenance": {"normalized_cuts": [LEFT_CUT, RIGHT_CUT], "raw_cuts": [left_raw, right_raw], "left_anchor": "Aidan delivers maps.", "right_anchor": "Spam's reviled, Nadia.", "retained_letters": len(normalize(retained)), "source_tape_retained_byte_for_byte": True},
            "whole_graft_equation": {"left_cursor_start": LEFT_CUT, "right_reverse_cursor_start": RIGHT_CUT - 1, "complete_left_event_tape": left_tape, "complete_right_obligation": right_tape[::-1], "equation_checked_after_authoring": True, "equation_holds": left_tape == right_tape[::-1], "incremental_residual_tracking": False, "cursor_updates_computed_during_search": False, "residual_owner": None, "grammar_slots_were_online": False},
            "bounded_attempts": attempts, "strict_global_checks": strict,
            "repair_debt": {"inherited_proper_spans": True, "inherited_repeated_scaffolding": True, "rough_syntax": True, "human_reader_validation": False},
        }],
    }


if __name__ == "__main__":
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    row = payload["rows"][0]
    print(json.dumps({"artifact": str(OUT.relative_to(ROOT)), "letters": row["independent_audit"]["normalized_letters"], "sha256": row["independent_audit"]["sha256_forward"]}, indent=2))
    print(row["rendered"])
