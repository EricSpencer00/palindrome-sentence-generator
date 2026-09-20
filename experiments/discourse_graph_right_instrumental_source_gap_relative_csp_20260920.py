"""Right-side instrumental/source-gap relative in the discourse-graph CSP.

This lane executes the next construction recorded by
``discourse-graph-right-benefactive-gap-relative-csp-20260920``.  It keeps the
preceding typed lexical bank unchanged and adds exactly one independently
authored graph edge: a right-clause source antecedent followed by a
``from whom`` relative whose source argument is a gap.  ``from`` is also
recorded as the instrumental/source case feature; it is not inferred from a
finished string.

Both clauses are emitted in ordinary English order from opposite character
frontiers.  A character equation is checked as soon as both characters are
exposed.  The lane never reverses a completed tape, repairs a failed sentence,
or scores an individual search with RLAIF.  Complete intact prose controls
are retained separately from exact closures.  Every rendering receives an
independent outside-in two-pointer audit, a forward/reverse SHA-256 audit,
and a post-enumeration recheck.  These programmatic checks screen shortcuts;
they do not certify human readability.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/discourse-graph-right-instrumental-source-gap-relative-csp-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"

EXPERIMENT_ID = "discourse-graph-right-instrumental-source-gap-relative-csp-20260920"
NOVELTY_SIGNATURE = (
    "typed-bilateral-discourse-graph|instrumental-source-gap-relative-attachment|"
    "source-antecedent-role|from-whom|live-character-equations"
)


def letters(text: str) -> str:
    """Return the normalized letter tape used by the exact definition."""

    return re.sub(r"[^a-z]", "", text.casefold())


def two_pointer_audit(text: str) -> dict[str, Any]:
    """Audit exactness with a fresh outside-in pointer walk."""

    tape = letters(text)
    mismatch = None
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            mismatch = {
                "offset": left,
                "left": tape[left],
                "right": tape[right],
            }
            break
        left += 1
        right -= 1
    return {
        "letters": len(tape),
        "normalized_tape": tape,
        "pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "audit_implementation": "independent outside-in two-pointer scan",
    }


def sha_audit(text: str) -> dict[str, Any]:
    """Audit exactness independently through forward/reverse SHA-256."""

    tape = letters(text)
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    return {
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
        "audit_implementation": "independent forward/reverse SHA-256 comparison",
    }


def postrun_recheck(text: str) -> dict[str, Any]:
    """Recheck a rendering with separate indexing and digest expressions."""

    tape = letters(text)
    mismatch = next(
        (
            {
                "offset": index,
                "left": tape[index],
                "right": tape[-1 - index],
            }
            for index in range(len(tape) // 2)
            if tape[index] != tape[-1 - index]
        ),
        None,
    )
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256("".join(reversed(tape)).encode("ascii")).hexdigest()
    return {
        "letters": len(tape),
        "pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
        "mechanically_exact": bool(tape) and mismatch is None and forward == reverse,
        "audit_implementation": "post-run independent index walk plus reversed digest",
    }


def audit_rendering(text: str) -> dict[str, Any]:
    pointer = two_pointer_audit(text)
    digest = sha_audit(text)
    return {
        **pointer,
        **digest,
        "mechanically_exact": bool(pointer["pointer_exact"] and digest["sha_equal"]),
    }


# This bank is copied byte-for-byte in content from the preceding
# right-benefactive-gap lane.  The only new lexical material is the authored
# function marker ``from whom`` required by the new source-gap operator.
SUBJECTS: dict[str, tuple[str, ...]] = {
    "sg": ("the pilot", "a keeper", "the poet", "a sailor"),
    "pl": ("the pilots", "two keepers", "the poets", "three sailors"),
}

RECIPIENTS: dict[str, tuple[str, ...]] = {
    "sg": ("a keeper", "the poet", "a sailor"),
    "pl": ("two keepers", "the poets", "three sailors"),
}

ATTACHMENTS: dict[str, tuple[str, ...]] = {
    "event": ("at dawn", "after rain", "in haste"),
    "theme": ("near the quay", "by the gate", "under the arch"),
}

VERBS: dict[str, dict[str, dict[str, str]]] = {
    "map": {
        "present": {"sg": "maps", "pl": "map"},
        "past": {"sg": "mapped", "pl": "mapped"},
    },
    "log": {
        "present": {"sg": "logs", "pl": "log"},
        "past": {"sg": "logged", "pl": "logged"},
    },
}


# One new graph edge.  The explicit right ``from`` phrase is the source
# antecedent; the relative has no overt source complement after ``from whom``.
# Its subject, finite verb, and object remain fully overt, so it is a complete
# English relative rather than a fragment.
SOURCE_GAP_EDGE: dict[str, Any] = {
    "id": "source-observation-evidence",
    "relation": "instrumental_source",
    "connector": "as",
    "left_event": "map",
    "right_event": "log",
    "left_objects": ("a chart", "the cove", "the inlet"),
    "right_objects": ("a note", "the chart", "the signal"),
    "number_pairs": (("sg", "pl"), ("pl", "sg")),
    "tense_pairs": (("present", "past"), ("past", "present")),
    "attachment_pairs": (("event", "theme"), ("theme", "event")),
    "relative_attachment": {
        "side": "right",
        "site": "source",
        "antecedent_role": "source",
        "antecedent_feature": "right_source_antecedent_role",
        "case_feature": "instrumental_source_from",
        "relative_role": "source_gap",
        "relative_marker": "from whom",
        "source_marker": "from",
        "relative_event": "map",
        "relative_subject_feature": "right_relative_subject_number",
        "relative_object_source": "right_objects",
    },
}


@dataclass(frozen=True)
class Slot:
    role: str
    values: tuple[str, ...]


def state_rows() -> Iterable[dict[str, Any]]:
    """Enumerate the typed relation/tense/agreement product exactly once."""

    edge = SOURCE_GAP_EDGE
    relative = edge["relative_attachment"]
    for left_number, right_number in edge["number_pairs"]:
        for left_tense, right_tense in edge["tense_pairs"]:
            for left_attachment, right_attachment in edge["attachment_pairs"]:
                yield {
                    "edge_id": edge["id"],
                    "relation": edge["relation"],
                    "connector": edge["connector"],
                    "left_event": edge["left_event"],
                    "right_event": edge["right_event"],
                    "left_objects": edge["left_objects"],
                    "right_objects": edge["right_objects"],
                    "left_number": left_number,
                    "right_number": right_number,
                    "left_tense": left_tense,
                    "right_tense": right_tense,
                    "left_attachment": left_attachment,
                    "right_attachment": right_attachment,
                    "source_antecedent_feature": {
                        "feature_name": relative["antecedent_feature"],
                        "case_feature": relative["case_feature"],
                        "side": relative["side"],
                        "slot": relative["site"],
                        "role": relative["antecedent_role"],
                        "number": right_number,
                        "relative_role": relative["relative_role"],
                    },
                    "source_number": right_number,
                    "relative_marker": relative["relative_marker"],
                    "source_marker": relative["source_marker"],
                    "relative_event": relative["relative_event"],
                    "relative_subject_number": right_number,
                    "relative_tense": right_tense,
                    "relative_object_source": relative["relative_object_source"],
                }


def build_slots(state: dict[str, Any], side: str) -> tuple[Slot, ...]:
    """Build ordinary-order slots, with one source-gap relative on the right."""

    number = state[f"{side}_number"]
    tense = state[f"{side}_tense"]
    event = state[f"{side}_event"]
    attachment = state[f"{side}_attachment"]
    objects = tuple(state[f"{side}_objects"])
    finite_verb = VERBS[event][tense][number]

    if side == "left":
        return (
            Slot("subject", SUBJECTS[number]),
            Slot("finite_verb", (finite_verb,)),
            Slot("object", objects),
            Slot("attachment", ATTACHMENTS[attachment]),
            Slot("discourse_connector", (state["connector"],)),
        )

    feature = state["source_antecedent_feature"]
    if (
        feature["role"] != "source"
        or feature["slot"] != "source"
        or feature["case_feature"] != "instrumental_source_from"
        or feature["number"] != state["source_number"]
    ):
        raise AssertionError("source antecedent feature was lost")

    relative_number = state["relative_subject_number"]
    relative_tense = state["relative_tense"]
    relative_verb = VERBS[state["relative_event"]][relative_tense][relative_number]
    relative_objects = tuple(state["right_objects"])
    return (
        Slot("subject", SUBJECTS[number]),
        Slot("finite_verb", (finite_verb,)),
        Slot("object", objects),
        Slot("source_marker", (state["source_marker"],)),
        Slot("source_antecedent", RECIPIENTS[state["source_number"]]),
        Slot("source_gap_marker", (state["relative_marker"],)),
        Slot("relative_subject", SUBJECTS[relative_number]),
        Slot("relative_finite_verb", (relative_verb,)),
        Slot("relative_object", relative_objects),
        Slot("attachment", ATTACHMENTS[attachment]),
    )


def render_selection(left: list[str], right: list[str]) -> str:
    """Render selected slots in ordinary prose order."""

    left_clause = " ".join(left[:-1])
    connector = left[-1]
    right_clause = " ".join(right)
    return f"{left_clause}, {connector} {right_clause}."


def lexical_units(text: str) -> list[str]:
    return [letters(word) for word in re.findall(r"[A-Za-z]+", text)]


STOP_UNITS = {
    "a",
    "an",
    "the",
    "as",
    "from",
    "whom",
    "at",
    "after",
    "in",
    "near",
    "by",
    "under",
}


def provenance_flags(text: str, left: list[str], right: list[str]) -> dict[str, Any]:
    """Apply transparent shortcut flags to one complete rendering."""

    units = lexical_units(text)
    content = [unit for unit in units if unit not in STOP_UNITS]
    left_content = [
        unit for unit in lexical_units(" ".join(left[:-1])) if unit not in STOP_UNITS
    ]
    right_content = [
        unit for unit in lexical_units(" ".join(right)) if unit not in STOP_UNITS
    ]
    return {
        "clause_count": 2,
        "relative_clause_count": 1,
        "relative_is_instrumental_source_gap": True,
        "relative_has_overt_source": False,
        "source_antecedent_is_overt": True,
        "two_distinct_clauses": bool(left[:-1]) and bool(right),
        "complete_relative_predicate": len(right) >= 10,
        "nested_self_palindrome": any(
            len(unit) > 3 and unit == unit[::-1] for unit in content
        ),
        "repeated_units": len(content) != len(set(content)),
        "word_order_symmetry": units == list(reversed(units)),
        "mirrored_units": left_content == list(reversed(right_content)),
        "fragment": len(units) < 14,
        "catalogue_text": False,
        "borrowed_catalogue_text": False,
        "finished_tape_reversal": False,
        "posthoc_tape_edit": False,
        "posthoc_repair": False,
        "rlaif_per_search": False,
        "human_authored_lexicon": True,
        "independently_authored_instrumental_source_gap_edge": True,
    }


def row_from_selection(
    state: dict[str, Any],
    left: list[str],
    right: list[str],
    *,
    source: str,
    matched_prefix: int | None = None,
) -> dict[str, Any]:
    """Record a complete rendering with provenance and independent audits."""

    rendered = render_selection(left, right)
    audit = audit_rendering(rendered)
    recheck = postrun_recheck(rendered)
    flags = provenance_flags(rendered, left, right)
    forbidden = (
        "nested_self_palindrome",
        "repeated_units",
        "word_order_symmetry",
        "mirrored_units",
        "fragment",
        "catalogue_text",
        "borrowed_catalogue_text",
        "finished_tape_reversal",
        "posthoc_tape_edit",
        "posthoc_repair",
        "rlaif_per_search",
    )
    clean = not any(flags[key] for key in forbidden)
    feature = state["source_antecedent_feature"]
    audit_agrees = audit["mechanically_exact"] == recheck["mechanically_exact"]
    return {
        "rendered": rendered,
        "length": audit["letters"],
        "state": state,
        "left_slots_normal_order": left,
        "right_slots_normal_order": right,
        "source": source,
        "matched_prefix_before_failure": matched_prefix,
        "audit": audit,
        "independent_postrun_recheck": recheck,
        "independent_audits_agree": audit_agrees,
        "provenance": {
            **flags,
            "graph_edge": state["edge_id"],
            "discourse_relation": state["relation"],
            "surface_connector": state["connector"],
            "tense_state": (state["left_tense"], state["right_tense"]),
            "agreement_state": (state["left_number"], state["right_number"]),
            "attachment_state": (
                state["left_attachment"],
                state["right_attachment"],
            ),
            "relative_attachment": {
                "site": feature["slot"],
                "antecedent_role": feature["role"],
                "antecedent_feature": feature["feature_name"],
                "case_feature": feature["case_feature"],
                "antecedent_number": feature["number"],
                "relative_role": feature["relative_role"],
                "source_marker": state["source_marker"],
                "relative_marker": state["relative_marker"],
                "relative_subject_number": state["relative_subject_number"],
                "relative_tense": state["relative_tense"],
                "source_head": right[4],
            },
            "lexical_choices_emitted_online": True,
            "independent_clause_authorship": True,
            "prior_palindrome_seed_used": False,
            "lexical_bank_widened": False,
            "reader_evidence": "not_run; programmatic checks do not certify readability",
            "programmatic_clean_for_reader_package": clean,
        },
        "reader_eligibility": {
            "eligible_for_blinded_package": bool(
                audit["mechanically_exact"] and clean and audit_agrees
            ),
            "human_readability_certified": False,
            "reason": (
                "requires intact-versus-shuffled blinded human reading; no study "
                "run in this lane"
            ),
        },
    }


def online_search(
    state: dict[str, Any],
    *,
    max_nodes: int = 30_000,
    max_exact_rows: int = 32,
) -> dict[str, Any]:
    """Emit both clauses online and compare opposing characters immediately."""

    left_slots = build_slots(state, "left")
    right_slots = build_slots(state, "right")
    left: list[str | None] = [None] * len(left_slots)
    right: list[str | None] = [None] * len(right_slots)
    exact_rows: list[dict[str, Any]] = []
    first_mismatches: list[dict[str, Any]] = []
    nodes = 0
    live_prunes = 0
    completions = 0
    truncated = False

    def remember_mismatch(
        reason: str,
        matched: int,
        left_char: str | None = None,
        right_char: str | None = None,
    ) -> None:
        if len(first_mismatches) < 8:
            first_mismatches.append(
                {
                    "reason": reason,
                    "matched_prefix_letters": matched,
                    "left_char": left_char,
                    "right_char": right_char,
                }
            )

    def walk(li: int, lo: int, ri: int, ro: int, matched: int) -> None:
        nonlocal nodes, live_prunes, completions, truncated
        nodes += 1
        if nodes > max_nodes:
            truncated = True
            return

        if li == len(left_slots) and ri < 0:
            completions += 1
            if len(exact_rows) < max_exact_rows:
                chosen_left = [value for value in left if value is not None]
                chosen_right = [value for value in right if value is not None]
                row = row_from_selection(
                    state,
                    chosen_left,
                    chosen_right,
                    source="online_exact_closure",
                    matched_prefix=matched,
                )
                if row["audit"]["mechanically_exact"]:
                    exact_rows.append(row)
            return

        if li == len(left_slots) or ri < 0:
            live_prunes += 1
            remember_mismatch("side_length_exhausted", matched)
            return

        if left[li] is not None:
            left_tape = letters(left[li] or "")
            if lo >= len(left_tape):
                walk(li + 1, 0, ri, ro, matched)
                return
        if right[ri] is not None:
            right_tape = letters(right[ri] or "")
            if ro < 0:
                walk(li, lo, ri - 1, 0, matched)
                return

        if left[li] is None:
            for value in left_slots[li].values:
                left[li] = value
                walk(li, 0, ri, ro, matched)
            left[li] = None
            return
        if right[ri] is None:
            for value in right_slots[ri].values:
                right[ri] = value
                walk(li, lo, ri, len(letters(value)) - 1, matched)
            right[ri] = None
            return

        left_tape = letters(left[li] or "")
        right_tape = letters(right[ri] or "")
        left_char = left_tape[lo]
        right_char = right_tape[ro]
        if left_char != right_char:
            live_prunes += 1
            remember_mismatch("character_equation", matched, left_char, right_char)
            return
        walk(li, lo + 1, ri, ro - 1, matched + 1)

    walk(0, 0, len(right_slots) - 1, 0, 0)
    return {
        "online_nodes": nodes,
        "live_character_prunes": live_prunes,
        "complete_closures": completions,
        "truncated": truncated,
        "first_mismatches": first_mismatches,
        "exact_rows": exact_rows,
    }


def controls_for_state(state: dict[str, Any], limit: int = 24) -> list[dict[str, Any]]:
    """Retain complete ordinary-prose controls from this typed edge."""

    left_slots = build_slots(state, "left")
    right_slots = build_slots(state, "right")
    left_products = list(itertools.product(*(slot.values for slot in left_slots)))
    right_products = list(itertools.product(*(slot.values for slot in right_slots)))
    rows: list[dict[str, Any]] = []
    for index in range(min(limit, len(left_products), len(right_products))):
        left_choice = list(left_products[index])
        right_choice = list(right_products[(index * 7 + 3) % len(right_products)])
        rows.append(
            row_from_selection(
                state,
                left_choice,
                right_choice,
                source="instrumental_source_gap_edge_complete_prose_control",
            )
        )
    return rows


def registry_preflight() -> dict[str, Any]:
    """Read the registry and reject an exact signature collision."""

    try:
        registry = json.loads(REGISTRY.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "unverified",
            "signature": NOVELTY_SIGNATURE,
            "reason": f"could not read current registry: {exc}",
        }
    entries: list[dict[str, Any]] = []
    for section in ("entries", "audit_reports", "excluded"):
        values = registry.get(section, [])
        if isinstance(values, list):
            entries.extend(value for value in values if isinstance(value, dict))
    collisions = [
        entry.get("id")
        for entry in entries
        if entry.get("signature") == NOVELTY_SIGNATURE
    ]
    return {
        "status": "passed" if not collisions else "collision",
        "signature": NOVELTY_SIGNATURE,
        "registry_version": registry.get("version"),
        "registry_entry_count": len(entries),
        "exact_signature_collisions": collisions,
        "nearest_existing_signatures": [
            "discourse-graph-object-gap-relative-csp-20260920",
            "discourse-graph-right-benefactive-gap-relative-csp-20260920",
            "semantic-role-instrumental-relative-diverse-csp-20260920",
        ],
        "endpoint_sweep_guard": {
            "new_endpoint": "right_source_antecedent|from_whom|instrumental_source_from",
            "prior_endpoint_reused": False,
            "note": (
                "This is one new typed source-role edge, not another lexical or "
                "an|an/ar|ar endpoint sweep."
            ),
        },
        "distinction": (
            "One independently authored right-side source antecedent is exposed "
            "with from, then licensed by a from-whom relative with an implicit "
            "source role.  The preceding lexical bank and relation/tense/number/"
            "attachment product remain unchanged; opposing character equations "
            "stay live."
        ),
    }


def run() -> dict[str, Any]:
    preflight = registry_preflight()
    if preflight["status"] != "passed":
        raise RuntimeError(f"novelty preflight rejected {EXPERIMENT_ID}: {preflight}")

    all_controls: list[dict[str, Any]] = []
    all_exact: list[dict[str, Any]] = []
    state_reports: list[dict[str, Any]] = []
    total_nodes = 0
    total_prunes = 0
    total_closures = 0
    for index, state in enumerate(state_rows(), start=1):
        search = online_search(state)
        controls = controls_for_state(state)
        all_controls.extend(controls)
        all_exact.extend(search["exact_rows"])
        total_nodes += search["online_nodes"]
        total_prunes += search["live_character_prunes"]
        total_closures += search["complete_closures"]
        state_reports.append(
            {
                "state_index": index,
                "state": state,
                "online": {
                    "nodes": search["online_nodes"],
                    "live_character_prunes": search["live_character_prunes"],
                    "complete_online_closures": search["complete_closures"],
                    "truncated": search["truncated"],
                    "first_mismatches": search["first_mismatches"],
                },
                "rendered_controls": len(controls),
            }
        )

    all_rows = all_controls + all_exact
    recheck_failures = [
        row["rendered"]
        for row in all_rows
        if not row["independent_audits_agree"]
        or row["audit"]["letters"] != row["independent_postrun_recheck"]["letters"]
    ]
    clean_exact = [
        row
        for row in all_exact
        if row["length"] > 38
        and row["provenance"]["programmatic_clean_for_reader_package"]
        and row["independent_audits_agree"]
    ]
    all_controls.sort(key=lambda row: (-row["length"], row["rendered"]))
    all_exact.sort(key=lambda row: (-row["length"], row["rendered"]))
    return {
        "experiment_id": EXPERIMENT_ID,
        "method": (
            "one independently authored right-side instrumental/source-gap "
            "relative edge in the bilateral discourse graph; an overt source "
            "antecedent with from licenses a from-whom relative whose source "
            "argument is implicit, while relation, tense, agreement, attachment, "
            "and opposing character equations remain live"
        ),
        "signature": NOVELTY_SIGNATURE,
        "stats": {
            "new_instrumental_source_gap_relative_edges": 1,
            "typed_discourse_states": len(state_reports),
            "online_nodes": total_nodes,
            "live_character_prunes": total_prunes,
            "complete_online_closures": total_closures,
            "rendered_complete_prose_controls": len(all_controls),
            "raw_exact_online_rows": len(all_exact),
            "exact_clean_above_38": len(clean_exact),
            "reader_worthy": 0,
            "max_rendered_control_letters": max(
                (row["length"] for row in all_controls), default=0
            ),
            "max_exact_letters": max((row["length"] for row in all_exact), default=0),
            "explicit_source_antecedent_feature_checks": len(state_reports),
            "source_relative_object_slots": 1,
            "independent_recheck_failures": len(recheck_failures),
            "lexical_bank_widened": False,
        },
        "exact_candidates": clean_exact,
        "raw_exact_rows": all_exact,
        "strongest_intact_controls": all_controls[:24],
        "state_reports": state_reports,
        "novelty_preflight": preflight,
        "provenance": {
            "lexicon": (
                "preceding right-benefactive-gap discourse-graph lexical bank "
                "copied unchanged; no content-word expansion"
            ),
            "graph": (
                "one independently authored source-observation-evidence edge only"
            ),
            "new_operator": (
                "right source antecedent with from followed by from whom + overt "
                "subject + finite verb + object, leaving the source role implicit"
            ),
            "emission": (
                "left slots forward and right slots from the right frontier; "
                "exposed characters compared immediately"
            ),
            "exact_audit": (
                "independent two-pointer scan plus forward/reverse SHA-256 on "
                "every rendered row, with a separate post-run recheck"
            ),
            "forbidden_shortcuts": [
                "finished-tape reversal",
                "posthoc repair",
                "mirrored or self-palindromic units",
                "word-order-only symmetry",
                "catalogue or borrowed sentence text",
                "fragments",
                "per-search RLAIF",
            ],
            "reader_gate": (
                "closed: no human study in this lane; complete controls are not "
                "readability certification"
            ),
        },
        "next_construction": (
            "Hold this source edge out and add one independently authored right-side "
            "instrumental with-which relative whose overt theme is the antecedent "
            "and whose instrument role is a new typed feature; preserve the same "
            "lexical bank, reject the from-whom replay, and keep live opposing "
            "character equations and complete prose controls."
        ),
        "status": (
            "no exact-clean closure above 38; complete instrumental/source-gap "
            "prose controls and independent mismatch certificates retained"
            if not clean_exact
            else "exact-clean closure above 38 found but still requires blinded human reading"
        ),
    }


if __name__ == "__main__":
    result = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
    for row in result["strongest_intact_controls"][:8]:
        print(row["length"], row["rendered"])
