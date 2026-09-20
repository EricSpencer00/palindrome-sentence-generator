"""Tiny human-authored discourse-graph search with a live bilateral CSP.

This lane is deliberately small and scene-first.  Each graph edge describes two
different clauses and carries a discourse relation, surface connective, tense,
subject number, and attachment type.  The left clause is emitted from its first
slot while the right clause is emitted from its last slot; characters are
compared immediately as the two lexical frontiers advance.  The right clause
is not made by reversing a completed tape: its slots are selected from the
right-hand frontier and later rendered in their ordinary order.

The experiment is a search-space probe, not a readability claim.  Every
rendered control is retained as ordinary two-clause prose, and every closure is
re-audited by an independent pointer scan and forward/reverse SHA-256 hashes.
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
OUT = ROOT / "runs/discourse-graph-bilateral-csp-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"

NOVELTY_SIGNATURE = (
    "human-authored-discourse-edge|two-clause-bilateral-online-csp|"
    "relation-tense-agreement-attachment-state|edge-conditioned-slots"
)


def letters(text: str) -> str:
    """Return the letter tape used by the exact palindrome definition."""

    return re.sub(r"[^a-z]", "", text.casefold())


def independent_audit(text: str) -> dict[str, Any]:
    """Audit a rendered sentence without calling the search routine."""

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
        "audit_implementation": "independent two-pointer scan plus independent forward/reverse SHA-256",
    }


@dataclass(frozen=True)
class Slot:
    role: str
    values: tuple[str, ...]


# These are deliberately hand-authored lexical banks.  They are not imported
# from the old seed, a corpus sentence, or an existing palindrome catalogue.
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
    "mark": {
        "present": {"sg": "marks", "pl": "mark"},
        "past": {"sg": "marked", "pl": "marked"},
    },
    "follow": {
        "present": {"sg": "follows", "pl": "follow"},
        "past": {"sg": "followed", "pl": "followed"},
    },
    "watch": {
        "present": {"sg": "watches", "pl": "watch"},
        "past": {"sg": "watched", "pl": "watched"},
    },
    "keep": {
        "present": {"sg": "keeps", "pl": "keep"},
        "past": {"sg": "kept", "pl": "kept"},
    },
}


# A graph edge is semantic, not a mirrored phrase pair.  The two event frames
# are intentionally different; each edge can vary tense, agreement, and which
# type of adjunct attaches to each event or theme.
GRAPH_EDGES: tuple[dict[str, Any], ...] = (
    {
        "id": "mapping-evidence",
        "relation": "evidence",
        "connector": "as",
        "left_event": "map",
        "right_event": "log",
        "left_objects": ("a chart", "the cove", "the inlet"),
        "right_objects": ("a note", "the chart", "the signal"),
        "number_pairs": (("sg", "sg"), ("pl", "pl")),
        "tense_pairs": (("present", "present"), ("past", "past")),
        "attachment_pairs": (("event", "theme"), ("theme", "event")),
    },
    {
        "id": "marking-sequence",
        "relation": "temporal-sequence",
        "connector": "while",
        "left_event": "mark",
        "right_event": "follow",
        "left_objects": ("the cove", "a chart", "the path"),
        "right_objects": ("the chart", "the path", "the inlet"),
        "number_pairs": (("sg", "pl"), ("pl", "sg")),
        "tense_pairs": (("past", "past"), ("present", "present")),
        "attachment_pairs": (("theme", "event"), ("event", "theme")),
    },
    {
        "id": "watching-consequence",
        "relation": "consequence",
        "connector": "so",
        "left_event": "watch",
        "right_event": "keep",
        "left_objects": ("the gate", "the harbor", "the signal"),
        "right_objects": ("a watch", "the chart", "the lantern"),
        "number_pairs": (("pl", "sg"), ("sg", "pl")),
        "tense_pairs": (("present", "present"), ("past", "past")),
        "attachment_pairs": (("event", "theme"), ("theme", "event")),
    },
)


def state_rows() -> Iterable[dict[str, Any]]:
    """Enumerate the tiny graph state product before lexical emission."""

    for edge in GRAPH_EDGES:
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
                    }


def build_slots(state: dict[str, Any], side: str) -> tuple[Slot, ...]:
    """Build one ordinary-order clause slot list for a graph state."""

    number = state[f"{side}_number"]
    tense = state[f"{side}_tense"]
    event = state[f"{side}_event"]
    attachment = state[f"{side}_attachment"]
    objects = tuple(state[f"{side}_objects"])
    subject_values = SUBJECTS[number]
    verb = VERBS[event][tense][number]
    slots = (
        Slot("subject", subject_values),
        Slot("finite_verb", (verb,)),
        Slot("object", objects),
        Slot("attachment", ATTACHMENTS[attachment]),
    )
    # The connective is emitted after the left clause, as part of the same
    # online left frontier.  It is not treated as a hidden center token.
    if side == "left":
        return slots + (Slot("discourse_connector", (state["connector"],)),)
    return slots


def render_selection(left: list[str], right: list[str]) -> str:
    """Render normal-order slots after online choices have been made."""

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
    "so",
    "yet",
    "while",
    "at",
    "after",
    "in",
    "near",
    "by",
    "under",
}


def provenance_flags(text: str, left: list[str], right: list[str]) -> dict[str, Any]:
    """Compute transparent shortcut flags for a rendered two-clause row."""

    all_units = lexical_units(text)
    content = [unit for unit in all_units if unit not in STOP_UNITS]
    left_content = [unit for unit in lexical_units(" ".join(left[:-1])) if unit not in STOP_UNITS]
    right_content = [unit for unit in lexical_units(" ".join(right)) if unit not in STOP_UNITS]
    return {
        "clause_count": 2,
        "two_distinct_clauses": bool(left[:-1]) and bool(right),
        "nested_self_palindrome": any(len(unit) > 3 and unit == unit[::-1] for unit in content),
        "repeated_units": len(content) != len(set(content)),
        "word_order_symmetry": all_units == list(reversed(all_units)),
        "mirrored_units": left_content == list(reversed(right_content)),
        "fragment": len(all_units) < 10,
        "catalogue_text": False,
        "borrowed_catalogue_text": False,
        "finished_tape_reversal": False,
        "posthoc_tape_edit": False,
        "posthoc_repair": False,
        "rlaif_per_search": False,
        "human_authored_lexicon": True,
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
            "lexical_choices_emitted_online": True,
            "independent_clause_authorship": True,
            "prior_palindrome_seed_used": False,
            "reader_evidence": "not_run; programmatic checks do not certify readability",
            "programmatic_clean_for_reader_package": clean,
        },
        "reader_eligibility": {
            "eligible_for_blinded_package": bool(audit["mechanically_exact"] and clean),
            "human_readability_certified": False,
            "reason": "requires intact-versus-shuffled blinded human reading; no study run in this lane",
        },
    }


def online_search(
    state: dict[str, Any],
    *,
    max_nodes: int = 20_000,
    max_exact_rows: int = 32,
) -> dict[str, Any]:
    """Emit both clause frontiers online and compare exposed characters."""

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

    def remember_mismatch(reason: str, matched: int, left_char: str | None = None, right_char: str | None = None) -> None:
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

        # Complete closure means every lexical slot on both sides was consumed.
        if li == len(left_slots) and ri < 0:
            completions += 1
            if len(exact_rows) < max_exact_rows:
                chosen_left = [value for value in left if value is not None]
                chosen_right = [value for value in right if value is not None]
                row = row_from_selection(state, chosen_left, chosen_right, source="online_exact_closure", matched_prefix=matched)
                if row["audit"]["mechanically_exact"]:
                    exact_rows.append(row)
            return

        # An exhausted side with live material on the other side cannot close.
        if li == len(left_slots) or ri < 0:
            live_prunes += 1
            remember_mismatch("side_length_exhausted", matched)
            return

        # Advance over a fully consumed lexical item without comparing a space.
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

        # Select lexical material only when that frontier reaches a fresh slot.
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
        next_lo = lo + 1
        next_ro = ro - 1
        walk(li, next_lo, ri, next_ro, matched + 1)

    walk(0, 0, len(right_slots) - 1, 0, 0)
    return {
        "online_nodes": nodes,
        "live_character_prunes": live_prunes,
        "complete_closures": completions,
        "truncated": truncated,
        "first_mismatches": first_mismatches,
        "exact_rows": exact_rows,
    }


def controls_for_state(state: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    """Render intact controls from the same graph state, without tape repair."""

    left_slots = build_slots(state, "left")
    right_slots = build_slots(state, "right")
    left_products = list(itertools.product(*(slot.values for slot in left_slots)))
    right_products = list(itertools.product(*(slot.values for slot in right_slots)))
    rows: list[dict[str, Any]] = []
    for index in range(min(limit, len(left_products), len(right_products))):
        # Different strides prevent the control list from being a word-order
        # mirror or a single repeated lexical pair.
        left_choice = list(left_products[index])
        right_choice = list(right_products[(index * 5 + 1) % len(right_products)])
        rows.append(row_from_selection(state, left_choice, right_choice, source="same_state_complete_prose_control"))
    return rows


def registry_preflight() -> dict[str, Any]:
    """Check the current registry before declaring the signature novel."""

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
    collisions = [entry.get("id") for entry in entries if entry.get("signature") == NOVELTY_SIGNATURE]
    nearest = [
        "discourse-graph-walk-palindrome-20260915",
        "role-labeled-grammar-csp-20260918",
        "discourse-conditioned-clause-csp-20260920",
    ]
    return {
        "status": "passed" if not collisions else "collision",
        "signature": NOVELTY_SIGNATURE,
        "registry_version": registry.get("version"),
        "registry_entry_count": len(entries),
        "exact_signature_collisions": collisions,
        "nearest_existing_signatures": nearest,
        "distinction": (
            "A tiny hand-authored two-node discourse edge carries relation, connective, tense, "
            "number agreement, and event/theme attachment while independently selecting the "
            "left frontier forward and the right clause frontier backward.  It emits complete "
            "ordinary-order clauses after the live character CSP, rather than walking a prebuilt "
            "event chain, reversing a tape, or repairing a finished sentence."
        ),
    }


def run() -> dict[str, Any]:
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
    longest_controls = all_controls[:24]
    return {
        "experiment_id": "discourse-graph-bilateral-csp-20260920",
        "method": (
            "tiny human-authored two-node discourse graph; edge-conditioned slots carry "
            "relation, tense, agreement, and event/theme attachment while left and right "
            "clause frontiers emit online under a bilateral character CSP"
        ),
        "signature": NOVELTY_SIGNATURE,
        "stats": {
            "graph_edges": len(GRAPH_EDGES),
            "typed_discourse_states": len(state_reports),
            "online_nodes": total_nodes,
            "live_character_prunes": total_prunes,
            "complete_online_closures": total_closures,
            "rendered_complete_prose_controls": len(all_controls),
            "raw_exact_online_rows": len(all_exact),
            "exact_clean_above_38": len(clean_exact),
            "reader_worthy": 0,
            "max_rendered_control_letters": max((row["length"] for row in all_controls), default=0),
            "max_exact_letters": max((row["length"] for row in all_exact), default=0),
        },
        "exact_candidates": clean_exact,
        "raw_exact_rows": all_exact,
        "strongest_intact_controls": longest_controls,
        "state_reports": state_reports,
        "novelty_preflight": registry_preflight(),
        "provenance": {
            "lexicon": "small hand-authored subject, event, object, and attachment banks",
            "graph": "three independently authored two-clause discourse edges",
            "emission": "left slots forward and right slots from the right frontier; characters compared immediately",
            "right_clause_rendering": "selected normal-order slots rendered after closure; no completed-tape reversal",
            "exact_audit": "independent two-pointer scan plus forward/reverse SHA-256 on every rendered row",
            "forbidden_shortcuts": [
                "finished-tape reversal",
                "posthoc repair",
                "mirrored or self-palindromic units",
                "word-order-only symmetry",
                "catalogue or borrowed sentence text",
                "per-search RLAIF or reward ranking",
            ],
            "reader_gate": "closed: no human study in this lane; controls are not readability certification",
        },
        "next_construction": (
            "Add one authored relative-clause attachment edge with an explicit antecedent "
            "feature, keeping the same live two-frontier character equations and agreement "
            "state; do not widen the lexical bank or add repair after a mismatch."
        ),
        "status": (
            "no exact-clean closure above 38; complete two-clause controls and mismatch "
            "certificates retained"
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
