"""Object-gap relative attachment in the bilateral discourse-graph CSP.

This is the next constructive operator after
``discourse_graph_relative_attachment_csp_20260920``.  It adds exactly one
new edge: an independently authored relative clause attached to the *theme*
object of the left event.  The relative clause has an overt subject and finite
verb but no overt object; the left theme is its object gap.  The lexical bank
is copied unchanged from the preceding lane.

The search emits ordinary-order slots from opposite character frontiers.  A
character equation is checked as soon as both frontiers expose that
character; no completed tape is reversed and no failed rendering is repaired.
Every complete control and every closure candidate receives an independent
two-pointer audit and forward/reverse SHA-256 audit.  Programmatic flags only
screen shortcuts; they do not certify human readability.
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
OUT = ROOT / "runs/discourse-graph-object-gap-relative-csp-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"

EXPERIMENT_ID = "discourse-graph-object-gap-relative-csp-20260920"
NOVELTY_SIGNATURE = (
    "typed-bilateral-discourse-graph|object-gap-relative-attachment-edge|"
    "theme-antecedent-role|live-character-equations"
)


def letters(text: str) -> str:
    """Return the letter tape used by the exact definition."""

    return re.sub(r"[^a-z]", "", text.casefold())


def independent_audit(text: str) -> dict[str, Any]:
    """Audit a rendering independently of the online search state."""

    tape = letters(text)
    mismatch = None
    for left in range(len(tape) // 2):
        right = len(tape) - 1 - left
        if tape[left] != tape[right]:
            mismatch = {
                "offset": left,
                "left": tape[left],
                "right": tape[right],
            }
            break
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    pointer_exact = bool(tape) and mismatch is None
    return {
        "letters": len(tape),
        "normalized_tape": tape,
        "pointer_exact": pointer_exact,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
        "mechanically_exact": pointer_exact and forward == reverse,
        "first_mismatch": mismatch,
        "audit_implementation": (
            "independent two-pointer scan plus independent forward/reverse SHA-256"
        ),
    }


# This lexical inventory is intentionally identical to the preceding
# discourse-graph relative lane.  Only the graph operator and its attachment
# feature change in this experiment.
SUBJECTS: dict[str, tuple[str, ...]] = {
    "sg": ("the pilot", "a keeper", "the poet", "a sailor"),
    "pl": ("the pilots", "two keepers", "the poets", "three sailors"),
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
    "follow": {
        "present": {"sg": "follows", "pl": "follow"},
        "past": {"sg": "followed", "pl": "followed"},
    },
}


# Exactly one new edge.  Its left theme/object is the antecedent of an
# object-gap relative.  The relative subject uses the existing left-agent
# number feature and its finite verb uses the existing left-event tense
# feature, so no hidden morphology or new lexical bank is introduced.
OBJECT_GAP_EDGE: dict[str, Any] = {
    "id": "object-observation-evidence",
    "relation": "evidence",
    "connector": "as",
    "left_event": "map",
    "right_event": "log",
    "left_objects": ("a chart", "the cove", "the inlet"),
    "right_objects": ("a note", "the chart", "the signal"),
    "number_pairs": (("sg", "pl"), ("pl", "sg")),
    "tense_pairs": (("present", "past"), ("past", "present")),
    "attachment_pairs": (("event", "theme"), ("theme", "event")),
    "relative_attachment": {
        "side": "left",
        "site": "object",
        "antecedent_role": "theme",
        "antecedent_feature": "left_theme_role",
        "relative_role": "object_gap",
        "relative_marker": "that",
        "relative_event": "follow",
        "relative_subject_feature": "left_agent_number",
        "relative_tense_feature": "left_event_tense",
    },
}


@dataclass(frozen=True)
class Slot:
    role: str
    values: tuple[str, ...]


def state_rows() -> Iterable[dict[str, Any]]:
    """Enumerate the new edge's typed relation/tense/agreement product."""

    edge = OBJECT_GAP_EDGE
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
                    "left_number": left_number,
                    "right_number": right_number,
                    "left_tense": left_tense,
                    "right_tense": right_tense,
                    "left_attachment": left_attachment,
                    "right_attachment": right_attachment,
                    "left_objects": edge["left_objects"],
                    "right_objects": edge["right_objects"],
                    "antecedent_feature": {
                        "feature_name": relative["antecedent_feature"],
                        "side": relative["side"],
                        "slot": relative["site"],
                        "role": relative["antecedent_role"],
                        "number": "sg",
                        "relative_role": relative["relative_role"],
                    },
                    "relative_marker": relative["relative_marker"],
                    "relative_event": relative["relative_event"],
                    "relative_subject_number": left_number,
                    "relative_tense": left_tense,
                }


def build_slots(state: dict[str, Any], side: str) -> tuple[Slot, ...]:
    """Build ordinary-order slots, with one object-gap relative on the left."""

    number = state[f"{side}_number"]
    tense = state[f"{side}_tense"]
    event = state[f"{side}_event"]
    attachment = state[f"{side}_attachment"]
    objects = tuple(state[f"{side}_objects"])
    finite_verb = VERBS[event][tense][number]

    if side == "left":
        relative = state["antecedent_feature"]
        if relative["role"] != "theme" or relative["slot"] != "object":
            raise AssertionError("object-gap antecedent role was not preserved")
        relative_subject = SUBJECTS[state["relative_subject_number"]]
        relative_verb = VERBS[state["relative_event"]][state["relative_tense"]][
            state["relative_subject_number"]
        ]
        # The object-gap relative has no object slot.  The left object is the
        # antecedent and is therefore followed by “that + subject + verb”.
        return (
            Slot("subject", SUBJECTS[number]),
            Slot("finite_verb", (finite_verb,)),
            Slot("antecedent_theme", objects),
            Slot("relative_marker", (state["relative_marker"],)),
            Slot("relative_subject", relative_subject),
            Slot("relative_finite_verb", (relative_verb,)),
            Slot("attachment", ATTACHMENTS[attachment]),
            Slot("discourse_connector", (state["connector"],)),
        )

    return (
        Slot("subject", SUBJECTS[number]),
        Slot("finite_verb", (finite_verb,)),
        Slot("object", objects),
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
    "that",
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
        unit
        for unit in lexical_units(" ".join(left[:-1]))
        if unit not in STOP_UNITS
    ]
    right_content = [unit for unit in lexical_units(" ".join(right)) if unit not in STOP_UNITS]
    return {
        "clause_count": 2,
        "relative_clause_count": 1,
        "relative_is_object_gap": True,
        "relative_has_overt_object": False,
        "two_distinct_clauses": bool(left[:-1]) and bool(right),
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
        "independently_authored_object_gap_edge": True,
    }


def row_from_selection(
    state: dict[str, Any],
    left: list[str],
    right: list[str],
    *,
    source: str,
    matched_prefix: int | None = None,
) -> dict[str, Any]:
    rendered = render_selection(left, right)
    audit = independent_audit(rendered)
    flags = provenance_flags(rendered, left, right)
    clean = not any(
        flags[key]
        for key in (
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
    )
    antecedent = state["antecedent_feature"]
    return {
        "rendered": rendered,
        "length": audit["letters"],
        "state": state,
        "left_slots_normal_order": left,
        "right_slots_normal_order": right,
        "source": source,
        "matched_prefix_before_failure": matched_prefix,
        "audit": audit,
        "provenance": {
            **flags,
            "graph_edge": state["edge_id"],
            "discourse_relation": state["relation"],
            "surface_connector": state["connector"],
            "tense_state": (state["left_tense"], state["right_tense"]),
            "agreement_state": (state["left_number"], state["right_number"]),
            "attachment_state": (state["left_attachment"], state["right_attachment"]),
            "relative_attachment": {
                "site": antecedent["slot"],
                "antecedent_role": antecedent["role"],
                "antecedent_feature": antecedent["feature_name"],
                "antecedent_number": antecedent["number"],
                "relative_role": antecedent["relative_role"],
                "relative_marker": state["relative_marker"],
                "relative_subject_number": state["relative_subject_number"],
                "relative_tense": state["relative_tense"],
                "antecedent_head": left[2],
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
                audit["mechanically_exact"] and clean
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
    """Emit both clauses online and compare exposed characters immediately."""

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
        if len(first_mismatches) >= 8:
            return
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
    """Retain complete ordinary-prose controls from this object-gap edge."""

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
                source="object_gap_edge_complete_prose_control",
            )
        )
    return rows


def registry_preflight() -> dict[str, Any]:
    """Inspect the registry and reject this run if the signature already exists."""

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
            "discourse-graph-bilateral-csp-20260920",
            "discourse-graph-relative-attachment-csp-20260920",
            "unequal-center-ditransitive-relative-complement-20260920",
        ],
        "distinction": (
            "One independently authored left-theme object-gap relative edge uses "
            "an overt relative subject and finite predicate with no overt relative "
            "object.  The preceding lexical bank is unchanged.  Its distinct "
            "theme antecedent-role feature is carried while relation, tense, "
            "agreement, attachment, and bilateral character equations stay live."
        ),
    }


def run() -> dict[str, Any]:
    preflight = registry_preflight()
    if preflight["status"] != "passed":
        raise RuntimeError(
            f"novelty preflight rejected {EXPERIMENT_ID}: {preflight}"
        )

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
                    "complete_closures": search["complete_closures"],
                    "truncated": search["truncated"],
                    "first_mismatches": search["first_mismatches"],
                },
                "rendered_controls": len(controls),
            }
        )

    clean_exact = [
        row
        for row in all_exact
        if row["length"] > 38
        and row["provenance"]["programmatic_clean_for_reader_package"]
    ]
    all_controls.sort(key=lambda row: (-row["length"], row["rendered"]))
    all_exact.sort(key=lambda row: (-row["length"], row["rendered"]))
    return {
        "experiment_id": EXPERIMENT_ID,
        "method": (
            "one independently authored left-theme object-gap relative edge in the "
            "bilateral discourse graph; an explicit theme antecedent-role feature "
            "attaches that to an overt relative subject and finite verb with no "
            "relative object, while relation, tense, agreement, attachment, and "
            "two-frontier character equations remain live"
        ),
        "signature": NOVELTY_SIGNATURE,
        "stats": {
            "new_object_gap_relative_edges": 1,
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
            "explicit_theme_antecedent_feature_checks": len(state_reports),
            "relative_object_slots": 0,
        },
        "exact_candidates": clean_exact,
        "raw_exact_rows": all_exact,
        "strongest_intact_controls": all_controls[:24],
        "state_reports": state_reports,
        "novelty_preflight": preflight,
        "provenance": {
            "lexicon": (
                "preceding discourse-graph-relative lexical bank copied unchanged; "
                "no content-word expansion"
            ),
            "graph": "one independently authored object-observation-evidence edge only",
            "new_operator": (
                "left theme/object antecedent with an object-gap relative: that + "
                "overt subject + finite verb, and no overt relative object"
            ),
            "emission": (
                "left slots forward and right slots from the right frontier; exposed "
                "characters compared immediately"
            ),
            "exact_audit": (
                "independent two-pointer scan plus forward/reverse SHA-256 on every "
                "rendered row"
            ),
            "forbidden_shortcuts": [
                "finished-tape reversal",
                "posthoc repair",
                "mirrored or self-palindromic units",
                "word-order-only symmetry",
                "catalogue or borrowed sentence text",
                "per-search RLAIF",
                "overt relative object in the object-gap clause",
            ],
            "reader_gate": (
                "closed: no human study in this lane; controls are not readability "
                "certification"
            ),
        },
        "next_construction": (
            "Add one independently authored right-side benefactive-gap relative "
            "edge with a distinct recipient antecedent-role feature, retaining the "
            "same unchanged lexical bank and live relation/tense/agreement character "
            "equations; do not replay this edge or add repair."
        ),
        "status": (
            "no exact-clean closure above 38; complete object-gap prose controls and "
            "mismatch certificates retained"
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
