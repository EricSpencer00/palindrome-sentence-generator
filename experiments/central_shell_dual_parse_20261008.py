"""Bounded dual-parse search against the frozen 650-letter seam parent.

This is a seam-local construction probe. It does not claim the inherited
parent is reader eligible: the untouched tape already fails global
distinct-content, repeated-unit, and proper-palindrome-span checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (
    has_distinct_content_words,
    has_repeated_nontrivial_unit,
    has_self_palindromic_proper_multiword_span,
    is_boundary_aligned_word_mirror,
    normalize_letters,
    phrasewise_reverse_boundary_offsets,
    tokenize,
)
from llm_palindrome.dual_parse import letter_tape, word_boundaries, word_residual_search


DEFAULT_PARENT = ROOT / "runs" / "incumbent-608-repeated-shell-repair-20261002.json"
DEFAULT_OUT = ROOT / "runs" / "central-shell-dual-parse-20261008.json"
PARENT_ID = "double-event-shell-repair-650"
PARENT_SHA256 = "2bd92686cbd01945be3869fbeee7ae9415cfcdae5cd616b26d89ec4f0acc54a9"

# The 36-letter-per-side central shell is the exact repeated construction to
# replace. These two surface halves reverse character-for-character.
OLD_LEFT = "Nadia delivers maps. Leon. Ari delivers maps."
OLD_RIGHT = "Spam's reviled, Ira. Noel; spam's reviled, Aidan."
OLD_LEFT_SPAN = (289, 325)  # normalized, half-open
OLD_RIGHT_SPAN = (325, 361)

# Frozen, deliberately small lexical/frame inventory. Every clause is an
# ordinary active transitive frame. Actors are names already present in the
# parent; action/object combinations are fixed before search. The search may
# repeat a participant as a discourse link, but not a verb or object.
ACTORS = ("Mara", "Nora", "Aidan", "Aron", "Tara", "Leon")
DETERMINERS = ("a", "the")
EVENT_FRAMES = {
    "carries": ("lamp", "letter"),
    "cleans": ("boat", "window"),
    "finds": ("key", "coin"),
    "folds": ("map", "cloth"),
    "guides": ("child", "guest"),
    "marks": ("page", "wall"),
    "opens": ("box", "door"),
    "paints": ("boat", "wall"),
    "reads": ("book", "note"),
    "repairs": ("roof", "gate"),
    "saves": ("bird", "guest"),
    "writes": ("report", "letter"),
}
FUNCTION_WORDS = frozenset({"a", "the"})
CLAUSES_PER_SIDE = 3
MAX_STATES = 500_000
MAX_RESULTS = 5_000
MIN_GROWTH_LETTERS = 20  # total full-tape growth above the exact parent


def _slots() -> tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]:
    slots = []
    for clause_index in range(CLAUSES_PER_SIDE):
        alternatives = tuple(
            f"{actor} {verb} {determiner} {patient}."
            for actor in ACTORS
            for verb, patients in EVENT_FRAMES.items()
            for patient in patients
            for determiner in DETERMINERS
        )
        slots.append((f"event{clause_index + 1}", alternatives))
    result = tuple(slots)
    assert len(result) == CLAUSES_PER_SIDE
    return result


SLOTS = _slots()


def _events(clauses: tuple[str, ...]) -> list[dict[str, str]]:
    events = []
    for clause in clauses:
        match = re.fullmatch(r"([A-Za-z]+) ([a-z]+) (a|the) ([a-z]+)\.", clause)
        if not match:
            continue
        agent, verb, determiner, patient = match.groups()
        events.append({
            "agent": normalize_letters(agent).capitalize(),
            "verb": normalize_letters(verb),
            "determiner": normalize_letters(determiner),
            "patient": normalize_letters(patient),
        })
    return events


def _partial_gate(left: tuple[str, ...], right: tuple[str, ...], stats: Counter) -> bool:
    """Apply role, event novelty, discourse, and reverse-token gates online."""
    stats["partial_states_seen"] += 1
    left_events, right_events = _events(left), _events(right)
    all_events = left_events + right_events

    # Do not permit a whole content word to be paired with its reversal.
    # Participant names may recur as discourse links, but are still checked
    # for reversal against every other emitted content word.
    content = [
        normalize_letters(word)
        for word in tokenize(" ".join(left + right))
        if normalize_letters(word) not in FUNCTION_WORDS
    ]
    for i, word in enumerate(content):
        if word == word[::-1] or any(word == other[::-1] for other in content[i + 1:]):
            stats["reverse_word_pair_rejections"] += 1
            return False

    # Reject impossible or repeated event frames as soon as a clause closes.
    for event in all_events:
        if event["verb"] not in EVENT_FRAMES or event["patient"] not in EVENT_FRAMES[event["verb"]]:
            stats["invalid_frame_rejections"] += 1
            return False
    frames = [(event["verb"], event["patient"]) for event in all_events]
    if len(frames) != len(set(frames)):
        stats["repeated_event_rejections"] += 1
        return False
    verbs = [event["verb"] for event in all_events]
    patients = [event["patient"] for event in all_events]
    if len(verbs) != len(set(verbs)) or len(patients) != len(set(patients)):
        stats["repeated_lexeme_rejections"] += 1
        return False

    # Each parse describes a small connected task sequence: one participant
    # acts across its three clauses. The opposite parse has its own participant.
    for side_events in (left_events, right_events):
        agents = {event["agent"].casefold() for event in side_events}
        if len(agents) > 1:
            stats["discourse_link_rejections"] += 1
            return False
    return True


def _endpoint_preflight() -> dict[str, object]:
    """Exhaustively check the necessary first-agent/final-patient condition.

    The left parse must expose its first actor against the reversed tape of
    the right parse's final patient. Since the current grammar emits a whole
    clause as a slot, one exposed string must be a prefix of the other before
    the chart can make progress. This preflight enumerates lexical endpoints
    directly and does not call the search engine.
    """
    patients = tuple(sorted({patient for choices in EVENT_FRAMES.values() for patient in choices}))
    checks = []
    prefix_viable = []
    mismatch_positions: Counter[int] = Counter()
    for actor in ACTORS:
        left = letter_tape(actor)
        for patient in patients:
            right_reversed = letter_tape(patient)[::-1]
            compared = min(len(left), len(right_reversed))
            matched = 0
            while matched < compared and left[matched] == right_reversed[matched]:
                matched += 1
            compatible = matched == compared
            record = {
                "left_actor": actor,
                "right_final_patient": patient,
                "right_reversed_tape": right_reversed,
                "matched_prefix_letters": matched,
                "compared_letters": compared,
                "prefix_compatible": compatible,
            }
            if compatible:
                prefix_viable.append(record)
            else:
                record["first_mismatch_index_zero_based"] = matched
                record["left_expected"] = left[matched]
                record["right_expected"] = right_reversed[matched]
                mismatch_positions[matched + 1] += 1
            checks.append(record)
    return {
        "method": "independent exhaustive prefix consumption over frozen endpoint domains",
        "left_outer_domain": list(ACTORS),
        "right_final_patient_domain": list(patients),
        "endpoint_pair_count": len(checks),
        "prefix_compatible_pair_count": len(prefix_viable),
        "prefix_compatible_pairs": prefix_viable,
        "rejected_pair_count": len(checks) - len(prefix_viable),
        "first_mismatch_position_counts_one_based": {
            str(position): mismatch_positions[position]
            for position in sorted(mismatch_positions)
        },
        "closest_rejected_pairs": sorted(
            (item for item in checks if not item["prefix_compatible"]),
            key=lambda item: (-item["matched_prefix_letters"], item["left_actor"], item["right_final_patient"]),
        )[:12],
        "pair_checks": checks,
    }


def _audit_parent(path: Path) -> tuple[dict, str, str]:
    payload = json.loads(path.read_text())
    row = next(item for item in payload["rows"] if item.get("id") == PARENT_ID)
    rendered = str(row["rendered"])
    tape = letter_tape(rendered)
    digest = hashlib.sha256(tape.encode("ascii")).hexdigest()
    if len(tape) != 650 or digest != PARENT_SHA256 or tape != tape[::-1]:
        raise ValueError("input is not the frozen exact 650-letter parent")
    if row.get("audit", {}).get("sha256_forward") != PARENT_SHA256:
        raise ValueError("stored parent audit does not match the frozen hash")
    if rendered.count(OLD_LEFT + " " + OLD_RIGHT) != 1:
        raise ValueError("expected central shell is absent or non-unique")
    left_tape, right_tape = letter_tape(OLD_LEFT), letter_tape(OLD_RIGHT)
    if left_tape != right_tape[::-1] or len(left_tape) != 36:
        raise AssertionError("frozen shell equation changed")
    if tape[OLD_LEFT_SPAN[0]:OLD_LEFT_SPAN[1]] != left_tape:
        raise AssertionError("left normalized shell coordinates drifted")
    if tape[OLD_RIGHT_SPAN[0]:OLD_RIGHT_SPAN[1]] != right_tape:
        raise AssertionError("right normalized shell coordinates drifted")
    return payload, rendered, tape


def _strict_shell_admission(result: dict) -> dict[str, object]:
    left, right = result["left"], result["right"]
    left_tape, right_tape = letter_tape(left), letter_tape(right)
    units = tokenize(left + " " + right)
    offsets = phrasewise_reverse_boundary_offsets(left, right)
    left_breaks = word_boundaries(left)
    reflected_right = tuple(sorted(len(right_tape) - n for n in word_boundaries(right)))
    return {
        "exact_half_equation": left_tape == right_tape[::-1],
        "minimum_total_growth": 2 * len(left_tape) - 72 >= MIN_GROWTH_LETTERS,
        "different_word_segmentation": left_breaks != reflected_right,
        "no_shared_reflected_word_boundary": not offsets,
        "no_reversed_content_word_pair": not any(
            word == word[::-1] for word in map(normalize_letters, units)
            if normalize_letters(word) not in FUNCTION_WORDS
        ) and not any(
            a == b[::-1]
            for i, a in enumerate(
                normalize_letters(word) for word in units
                if normalize_letters(word) not in FUNCTION_WORDS
            )
            for b in list(
                normalize_letters(word) for word in units
                if normalize_letters(word) not in FUNCTION_WORDS
            )[i + 1:]
        ),
        "no_repeated_nontrivial_unit": not has_repeated_nontrivial_unit(units),
        "no_proper_palindromic_span_inside_shell": not has_self_palindromic_proper_multiword_span(units),
        "not_word_order_symmetry": not is_boundary_aligned_word_mirror(units),
        "left_word_boundaries": list(left_breaks),
        "reflected_right_word_boundaries": list(reflected_right),
        "shared_reflected_word_offsets": list(offsets),
    }


def run(parent_path: Path = DEFAULT_PARENT, *, max_states: int = MAX_STATES,
        max_results: int = MAX_RESULTS) -> dict:
    parent_payload, parent_rendered, parent_tape = _audit_parent(parent_path)
    parent_tokens = tokenize(parent_rendered)
    parent_global = {
        "normalized_letters": len(parent_tape),
        "sha256": PARENT_SHA256,
        "distinct_content_words": has_distinct_content_words(parent_tokens),
        "no_repeated_nontrivial_unit": not has_repeated_nontrivial_unit(parent_tokens),
        "no_proper_palindromic_span": not has_self_palindromic_proper_multiword_span(parent_tokens),
    }
    inherited_gates = {
        key: value for key, value in parent_global.items()
        if key not in {"normalized_letters", "sha256"}
    }
    if inherited_gates != {
        "distinct_content_words": False,
        "no_repeated_nontrivial_unit": False,
        "no_proper_palindromic_span": False,
    }:
        raise AssertionError("expected inherited anti-shortcut debt changed; review protocol")

    endpoint = _endpoint_preflight()
    if endpoint["prefix_compatible_pair_count"] == 0:
        return {
            "experiment_id": "central-shell-dual-parse-20261008",
            "method": "preflighted typed word-residual dual-parse over a frozen symmetric shell",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "status": "infeasible_before_search",
            "search_launched": False,
            "protocol": {
                "parent_artifact": str(parent_path),
                "parent_id": PARENT_ID,
                "parent_normalized_letters": len(parent_tape),
                "parent_normalized_sha256": PARENT_SHA256,
                "shell_left_normalized_span": list(OLD_LEFT_SPAN),
                "shell_right_normalized_span": list(OLD_RIGHT_SPAN),
                "max_states_not_used": max_states,
                "max_results_not_used": max_results,
                "frame_domain": {verb: list(objects) for verb, objects in EVENT_FRAMES.items()},
                "actors": list(ACTORS),
                "determiners": list(DETERMINERS),
                "clauses_per_side": CLAUSES_PER_SIDE,
            },
            "parent_global_gate": parent_global,
            "old_shell_diagnostic": {
                "exact_reverse_equation": True,
                "shared_reflected_word_offsets": list(phrasewise_reverse_boundary_offsets(OLD_LEFT, OLD_RIGHT)),
                "repeated_delivers_maps_left": 2,
                "repeated_spams_reviled_right": 2,
                "whole_word_reverse_pairs": [
                    ["Nadia", "Aidan"], ["delivers", "reviled"],
                    ["maps", "spam's"], ["Leon", "Noel"], ["Ari", "Ira"],
                ],
            },
            "endpoint_preflight": endpoint,
            "stats": {
                "states": 0,
                "transitions": 0,
                "exact_half_equations": 0,
                "qualifying_seam_children": 0,
            },
            "obstruction": (
                "Every current left outer actor/final right patient pair conflicts "
                "before either complete endpoint slot can be consumed. The configured "
                "word-residual chart is therefore not launched; this is endpoint "
                "infeasibility for the frozen domains, not a 500k-state negative search."
            ),
            "extension_proposal": {
                "kind": "change endpoint grammar topology to a live multiword NP owner",
                "witness": {
                    "left_outer_word": "Mara",
                    "right_final_phrase": "a ram",
                    "left_tape": "mara",
                    "right_reversed_tape": "mara",
                    "boundary_relation": "one left word matches a two-word right noun phrase; no content token is the reverse of another",
                    "possible_final_clause": "Tara sees a ram.",
                },
                "next_search": (
                    "Replace whole-clause endpoint slots with a grammar-state chart that "
                    "exposes the rightmost NP as separate determiner and noun choices, "
                    "carries their character residual into the next left word, and admits "
                    "the 'Mara' / 'a ram' owner witness before expanding inner events. "
                    "Freeze a fresh event-frame grammar and rerun the same global debt audit."
                ),
                "not_done": "The witness proves only endpoint compatibility; it is not a complete exact child or a readability result.",
            },
        }

    stats = Counter()
    search = word_residual_search(
        SLOTS, SLOTS,
        max_states=max_states,
        max_results=max_results,
        allow_partial=lambda left, right: _partial_gate(left, right, stats),
        reject_intermediate_closure=True,
    )

    candidates = []
    for result in search["results"]:
        local = _strict_shell_admission(result)
        new_tape = letter_tape(result["rendered"])
        if len(new_tape) % 2:
            continue
        full_tape = (
            parent_tape[:OLD_LEFT_SPAN[0]] + new_tape
            + parent_tape[OLD_RIGHT_SPAN[1]:]
        )
        full_sha = hashlib.sha256(full_tape.encode("ascii")).hexdigest()
        full_exact = full_tape == full_tape[::-1]
        growth = len(full_tape) - len(parent_tape)
        local["full_parent_exact"] = full_exact
        local["full_parent_growth_letters"] = growth
        local["full_parent_sha256"] = full_sha
        local["inherited_global_shortcut_debt_remains"] = not all(
            value for key, value in parent_global.items()
            if key not in {"normalized_letters", "sha256"}
        )
        candidates.append({
            **result,
            "strict_shell_admission": local,
            "qualifies_for_seam_frontier": all(
                value for key, value in local.items()
                if key not in {
                    "left_word_boundaries", "reflected_right_word_boundaries",
                    "shared_reflected_word_offsets", "full_parent_sha256",
                    "full_parent_growth_letters", "inherited_global_shortcut_debt_remains",
                }
            ) and full_exact and growth >= MIN_GROWTH_LETTERS,
        })

    admitted = [item for item in candidates if item["qualifies_for_seam_frontier"]]
    admitted.sort(key=lambda item: (-item["strict_shell_admission"]["full_parent_growth_letters"], item["rendered"]))
    parent_stats = parent_payload.get("stats", {})
    return {
        "experiment_id": "central-shell-dual-parse-20261008",
        "method": "bounded typed word-residual dual-parse over a frozen symmetric shell",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": {
            "parent_artifact": str(parent_path),
            "parent_id": PARENT_ID,
            "parent_normalized_letters": len(parent_tape),
            "parent_normalized_sha256": PARENT_SHA256,
            "parent_status_claimed_by_source": parent_stats.get("working_length_incumbent"),
            "shell_left_normalized_span": list(OLD_LEFT_SPAN),
            "shell_right_normalized_span": list(OLD_RIGHT_SPAN),
            "old_left": OLD_LEFT,
            "old_right": OLD_RIGHT,
            "old_shell_letters_per_side": 36,
            "minimum_total_growth": MIN_GROWTH_LETTERS,
            "clauses_per_side": CLAUSES_PER_SIDE,
            "max_states": max_states,
            "max_results": max_results,
            "reject_intermediate_closure": True,
            "frame_domain": {verb: list(objects) for verb, objects in EVENT_FRAMES.items()},
            "actors": list(ACTORS),
            "determiners": list(DETERMINERS),
            "participant_coreference_allowed": "same actor within each three-event parse",
            "freshness": "selected verbs/objects must be unique; event frames cannot repeat",
            "seed": None,
        },
        "parent_global_gate": parent_global,
        "endpoint_preflight": endpoint,
        "old_shell_diagnostic": {
            "exact_reverse_equation": True,
            "left_content_words": list(tokenize(OLD_LEFT)),
            "right_content_words": list(tokenize(OLD_RIGHT)),
            "shared_reflected_word_offsets": list(phrasewise_reverse_boundary_offsets(OLD_LEFT, OLD_RIGHT)),
            "repeated_delivers_maps_left": 2,
            "repeated_spams_reviled_right": 2,
            "whole_word_reverse_pairs": [
                ["Nadia", "Aidan"], ["delivers", "reviled"],
                ["maps", "spam's"], ["Leon", "Noel"], ["Ari", "Ira"],
            ],
        },
        "stats": {
            "states": search["states"],
            "transitions": search["transitions"],
            "cap_reached": search["cap_reached"],
            "exact_half_equations": len(search["results"]),
            "qualifying_seam_children": len(admitted),
            "intermediate_closure_rejections": search["intermediate_closure_rejections"],
            **dict(stats),
        },
        "deepest_frontiers": search["dead_frontiers"],
        "exact_candidates": candidates,
        "qualifying_seam_children": admitted,
        "interpretation": (
            "A qualifying seam child is an exact local construction result only. "
            "The unchanged parent already fails global distinct-content, repeated-unit, "
            "and proper-palindrome-span checks, so this run cannot promote a readable item."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-states", type=int, default=MAX_STATES)
    parser.add_argument("--max-results", type=int, default=MAX_RESULTS)
    args = parser.parse_args()
    payload = run(args.parent, max_states=args.max_states, max_results=args.max_results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
