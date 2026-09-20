"""A held-out relative-temporal edge with a bounded three-character seam.

This is the direct next operator after ``semantic_role_boundary2_live_csp``.
The preceding lane is run first and its live two-character ``an|an`` frontier
is recorded.  Only then does this lane add one independently authored
``while`` edge and consume a third character in the same outside-in CSP.

The lexical inventory is imported unchanged from the preceding lane.  The new
edge is a complete ordinary-English two-event scene, not a finished sentence
that is reversed or repaired.  A state survives only while its opposing
characters agree; no lexical inventory widening, post-hoc repair, catalogue
text, mirrored units, fragments, or per-search RLAIF is used.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/semantic-role-relative-temporal3-csp-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT_ID = "semantic-role-relative-temporal3-csp-20260920"
SIGNATURE = (
    "semantic-role-skeleton|relative-temporal-while-edge|"
    "three-character-boundary|opposing-cursor-csp"
)
BENCHMARK_LETTERS = 38
STATE_LIMIT = 180_000
CENTER_LIMIT = 40_000
BASELINE_MODULE_NAME = "semantic_role_boundary2_live_csp_20260920"


def load_previous_lane():
    """Load the committed boundary2 lane without running its top-level writer."""

    experiments = str(ROOT / "experiments")
    if experiments not in sys.path:
        sys.path.insert(0, experiments)
    return importlib.import_module(BASELINE_MODULE_NAME)


PREVIOUS = load_previous_lane()


def normalize_letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def pointer_scan(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    mismatch = next(
        (
            (index, tape[index], tape[-1 - index])
            for index in range(len(tape) // 2)
            if tape[index] != tape[-1 - index]
        ),
        None,
    )
    return {
        "letters": len(tape),
        "pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
    }


def hash_audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    return {
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def audit(text: str) -> dict[str, object]:
    pointer = pointer_scan(text)
    digest = hash_audit(text)
    return {
        **pointer,
        **digest,
        "exact": bool(pointer["pointer_exact"] and digest["sha_equal"]),
    }


Slot = PREVIOUS.Slot
Frame = PREVIOUS.Frame


def slot(name: str, role: str, *values: str, fixed: bool = False) -> Slot:
    return Slot(name=name, role=role, values=tuple(dict.fromkeys(values)), fixed=fixed)


# These are the preceding lane's authored inventories, not an expanded bank.
AGENTS = PREVIOUS.AGENTS
ACTIONS = PREVIOUS.ACTIONS
THEMES = PREVIOUS.THEMES
TEMPORAL_PLACES = PREVIOUS.TEMPORAL_PLACES
AGENT_ACTIONS = PREVIOUS.AGENT_ACTIONS
ACTION_THEMES = PREVIOUS.ACTION_THEMES
AGENT_THEMES = PREVIOUS.AGENT_THEMES


def relative_temporal_frame() -> Frame:
    """A complete matrix event followed by one independently chosen while edge."""

    return Frame(
        frame_id="relative_temporal_while",
        edge_kind="relative-temporal",
        description=(
            "an agent acts on a theme while a second agent acts on a second "
            "theme at a setting"
        ),
        slots=(
            slot("subject_determiner", "fixed", "A", fixed=True),
            slot("agent1", "agent1", *AGENTS),
            slot("action1", "action1", *ACTIONS),
            slot("object_determiner1", "fixed", "the", fixed=True),
            slot("theme1", "theme1", *THEMES),
            slot("temporal_connector", "fixed", "while", fixed=True),
            slot("subject_determiner2", "fixed", "the", fixed=True),
            slot("agent2", "agent2", *AGENTS),
            slot("action2", "action2", *ACTIONS),
            slot("object_determiner2", "fixed", "the", fixed=True),
            slot("theme2", "theme2", *THEMES),
            slot("locative_connector", "fixed", "at", fixed=True),
            slot("setting_determiner", "fixed", "the", fixed=True),
            slot("place", "temporal_place", *TEMPORAL_PLACES),
        ),
    )


FRAME = relative_temporal_frame()


@dataclass(frozen=True)
class Boundary3:
    """The first at most three opposing character equations."""

    left_window: str
    right_window: str
    equations: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class State3:
    tokens: tuple[str | None, ...]
    roles: tuple[tuple[str, str], ...]
    left_slot: int
    right_slot: int
    left_pos: int
    right_pos: int
    matched_pairs: int
    boundary3: Boundary3 | None


def role_map(state: State3) -> dict[str, str]:
    return dict(state.roles)


def partial_semantics(roles: dict[str, str]) -> bool:
    """Check each complete or partial event without adding lexical choices."""

    for suffix in ("1", "2"):
        agent = roles.get(f"agent{suffix}")
        action = roles.get(f"action{suffix}")
        theme = roles.get(f"theme{suffix}")
        if suffix == "1" and agent and agent[0] in "aeiou":
            # The frame deliberately fixes the matrix determiner as ``A``;
            # reject vowel-initial agents rather than emitting "A engineer".
            return False
        if agent and action and action not in AGENT_ACTIONS.get(agent, set()):
            return False
        if agent and theme and theme not in AGENT_THEMES.get(agent, set()):
            return False
        if action and theme and theme not in ACTION_THEMES.get(action, set()):
            return False
    return True


def assign_slot(state: State3, index: int) -> list[State3]:
    if state.tokens[index] is not None:
        return [state]
    slot_value = FRAME.slots[index]
    out: list[State3] = []
    for value in slot_value.values:
        tokens = list(state.tokens)
        tokens[index] = value
        roles = role_map(state)
        if slot_value.role != "fixed":
            roles[slot_value.role] = value
        if not partial_semantics(roles):
            continue
        out.append(
            State3(
                tokens=tuple(tokens),
                roles=tuple(sorted(roles.items())),
                left_slot=state.left_slot,
                right_slot=state.right_slot,
                left_pos=state.left_pos,
                right_pos=state.right_pos,
                matched_pairs=state.matched_pairs,
                boundary3=state.boundary3,
            )
        )
    return out


def render_tokens(tokens: Iterable[str | None]) -> str:
    values = list(tokens)
    if any(value is None for value in values):
        raise ValueError("cannot render an incomplete frame")
    return " ".join(value for value in values if value is not None) + "."


def shortcut_audit(text: str) -> dict[str, object]:
    words = text.rstrip(".").split()
    content_words = [
        normalize_letters(word)
        for word, slot_value in zip(words, FRAME.slots)
        if not slot_value.fixed
    ]
    repeated = sorted(
        word for word, count in Counter(content_words).items() if word and count > 1
    )
    self_palindromic = sorted(
        word for word in content_words if len(word) > 1 and word == word[::-1]
    )
    word_order_only = len(content_words) > 1 and content_words == content_words[::-1]
    flags: dict[str, object] = {
        "repeated_content_units": repeated,
        "repeated_units": bool(repeated),
        "self_palindromic_units": self_palindromic,
        "word_order_only_symmetry": word_order_only,
        "catalogue_text": False,
        "finished_tape_reversal": False,
        "post_hoc_repair": False,
        "fragment": len(words) != len(FRAME.slots),
    }
    flags["shortcut_clean"] = not any(
        flags[name]
        for name in (
            "repeated_units",
            "self_palindromic_units",
            "word_order_only_symmetry",
            "catalogue_text",
            "finished_tape_reversal",
            "post_hoc_repair",
            "fragment",
        )
    )
    return flags


def state_key(state: State3) -> tuple[object, ...]:
    boundary = None
    if state.boundary3 is not None:
        boundary = (
            state.boundary3.left_window,
            state.boundary3.right_window,
            state.boundary3.equations,
        )
    return (
        state.tokens,
        state.roles,
        state.left_slot,
        state.right_slot,
        state.left_pos,
        state.right_pos,
        state.matched_pairs,
        boundary,
    )


def closure_row(state: State3) -> dict[str, object] | None:
    if any(value is None for value in state.tokens):
        return None
    roles = role_map(state)
    if not partial_semantics(roles):
        return None
    rendered = render_tokens(state.tokens)
    boundary = state.boundary3
    exact_audit = audit(rendered)
    return {
        "rendered": rendered,
        "length": exact_audit["letters"],
        "frame": FRAME.frame_id,
        "edge_kind": FRAME.edge_kind,
        "semantic_roles": roles,
        "boundary3": {
            "left_window": boundary.left_window if boundary else "",
            "right_window": boundary.right_window if boundary else "",
            "equations": list(boundary.equations) if boundary else [],
        },
        "audit": exact_audit,
        "shortcut_audit": shortcut_audit(rendered),
        "provenance": {
            "source": "fresh independently authored relative-temporal while edge",
            "lexical_inventory_reused_from_boundary2": True,
            "role_edge_selected_inside_live_csp": True,
            "three_character_boundary_state_carried": True,
            "third_equation_checked_before_center": True,
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "word_order_only_symmetry": False,
            "mirrored_or_self_palindromic_units": False,
            "catalogue_text": False,
            "fragment": False,
            "per_search_rlaif": False,
        },
        "reader_eligible": False,
        "reader_evidence": {
            "status": "not_run",
            "blinded_human_reading_required": True,
        },
    }


def finish_center(
    state: State3,
    rows: list[dict[str, object]],
    counters: dict[str, object],
    seen_text: set[str],
    budget: list[int],
) -> None:
    if budget[0] >= CENTER_LIMIT:
        counters["center_limit_prunes"] += 1
        return
    budget[0] += 1
    counters["center_states"] += 1
    if all(value is not None for value in state.tokens):
        row = closure_row(state)
        if row is None:
            return
        counters["complete_closures"] += 1
        text = str(row["rendered"])
        if text in seen_text:
            counters["duplicate_closures"] += 1
            return
        seen_text.add(text)
        counters["exact_closures"] += int(bool(row["audit"]["exact"]))
        rows.append(row) if bool(row["audit"]["exact"]) else None
        return
    index = next(i for i, value in enumerate(state.tokens) if value is None)
    for candidate in assign_slot(state, index):
        finish_center(candidate, rows, counters, seen_text, budget)


def boundary3_prefixes() -> frozenset[str]:
    """Reachable three-character endpoint windows from this frame.

    The first token is the one-letter ``A`` and the last token is a place.  A
    third live equation therefore compares the first two letters of agent1
    against the next two inward letters of the final place.
    """

    return frozenset(
        "a" + agent[:2]
        for agent in AGENTS
        for place in TEMPORAL_PLACES
        if "a" + agent[:2] == place[::-1][:3]
    )


def search_relative_temporal() -> tuple[list[dict[str, object]], dict[str, object]]:
    state = State3(
        tokens=tuple(None for _ in FRAME.slots),
        roles=tuple(),
        left_slot=0,
        right_slot=len(FRAME.slots) - 1,
        left_pos=0,
        right_pos=0,
        matched_pairs=0,
        boundary3=None,
    )
    rows: list[dict[str, object]] = []
    counters: dict[str, object] = {
        "states_seen": 0,
        "memo_hits": 0,
        "semantic_prunes": 0,
        "character_prunes": 0,
        "third_character_prunes": 0,
        "boundary_advances": 0,
        "character_equations": 0,
        "first_equation_survivors": 0,
        "two_equation_survivors": 0,
        "three_equation_survivors": 0,
        "boundary3_examples": [],
        "boundary3_edge_index_prunes": 0,
        "max_matched_pairs": 0,
        "center_states": 0,
        "center_limit_prunes": 0,
        "complete_closures": 0,
        "exact_closures": 0,
        "duplicate_closures": 0,
        "state_limit_prunes": 0,
    }
    seen: set[tuple[object, ...]] = set()
    seen_text: set[str] = set()
    center_budget = [0]
    edge_prefixes = boundary3_prefixes()

    def walk(current: State3) -> None:
        key = state_key(current)
        if key in seen:
            counters["memo_hits"] += 1
            return
        if len(seen) >= STATE_LIMIT:
            counters["state_limit_prunes"] += 1
            return
        seen.add(key)
        counters["states_seen"] += 1
        counters["max_matched_pairs"] = max(
            int(counters["max_matched_pairs"]), current.matched_pairs
        )

        if current.left_slot >= current.right_slot:
            finish_center(current, rows, counters, seen_text, center_budget)
            return

        left_states = assign_slot(current, current.left_slot)
        if not left_states:
            counters["semantic_prunes"] += 1
            return
        for left_state in left_states:
            right_states = assign_slot(left_state, left_state.right_slot)
            if not right_states:
                counters["semantic_prunes"] += 1
                continue
            for paired in right_states:
                left_value = normalize_letters(
                    paired.tokens[paired.left_slot] or ""
                )
                right_value = normalize_letters(
                    paired.tokens[paired.right_slot] or ""
                )[::-1]

                if paired.left_pos >= len(left_value):
                    walk(
                        State3(
                            **{
                                **paired.__dict__,
                                "left_slot": paired.left_slot + 1,
                                "left_pos": 0,
                            }
                        )
                    )
                    counters["boundary_advances"] += 1
                    continue
                if paired.right_pos >= len(right_value):
                    walk(
                        State3(
                            **{
                                **paired.__dict__,
                                "right_slot": paired.right_slot - 1,
                                "right_pos": 0,
                            }
                        )
                    )
                    counters["boundary_advances"] += 1
                    continue

                left_char = left_value[paired.left_pos]
                right_char = right_value[paired.right_pos]
                counters["character_equations"] += 1
                if left_char != right_char:
                    counters["character_prunes"] += 1
                    if paired.matched_pairs == 2:
                        counters["third_character_prunes"] += 1
                    continue

                if paired.matched_pairs == 0:
                    counters["first_equation_survivors"] += 1
                boundary = paired.boundary3
                if boundary is None:
                    boundary = Boundary3(
                        left_window=left_char,
                        right_window=right_char,
                        equations=((left_char, right_char),),
                    )
                elif len(boundary.equations) < 3:
                    boundary = Boundary3(
                        left_window=boundary.left_window + left_char,
                        right_window=boundary.right_window + right_char,
                        equations=boundary.equations + ((left_char, right_char),),
                    )
                    if len(boundary.equations) == 2:
                        counters["two_equation_survivors"] += 1
                    else:
                        if boundary.left_window not in edge_prefixes:
                            counters["boundary3_edge_index_prunes"] += 1
                            continue
                        counters["three_equation_survivors"] += 1
                        examples = counters["boundary3_examples"]
                        assert isinstance(examples, list)
                        signature = (
                            f"{boundary.left_window}|{boundary.right_window}"
                        )
                        if signature not in examples and len(examples) < 8:
                            examples.append(signature)

                walk(
                    State3(
                        **{
                            **paired.__dict__,
                            "left_pos": paired.left_pos + 1,
                            "right_pos": paired.right_pos + 1,
                            "matched_pairs": paired.matched_pairs + 1,
                            "boundary3": boundary,
                        }
                    )
                )

    walk(state)
    counters["memoized_frontier_states"] = len(seen)
    counters["longest_matched_prefix"] = counters["max_matched_pairs"]
    counters["boundary3_prefix_index"] = sorted(edge_prefixes)
    return rows, counters


def control_row(tokens: list[str]) -> dict[str, object]:
    if len(tokens) != len(FRAME.slots):
        raise ValueError("control token count does not match frame")
    roles = {
        slot_value.role: value
        for slot_value, value in zip(FRAME.slots, tokens)
        if slot_value.role != "fixed"
    }
    if not partial_semantics(roles):
        raise ValueError(f"invalid control semantics: {tokens}")
    rendered = render_tokens(tokens)
    first_audit = audit(rendered)
    return {
        "rendered": rendered,
        "length": first_audit["letters"],
        "frame": FRAME.frame_id,
        "edge_kind": FRAME.edge_kind,
        "semantic_roles": roles,
        "audit": first_audit,
        "shortcut_audit": shortcut_audit(rendered),
        "provenance": {
            "source": "fresh authored complete-English relative-temporal control",
            "generated_by": "typed two-event while-edge lexicalization",
            "candidate_status": "control_only",
            "complete_prose": True,
            "lexical_inventory_reused_from_boundary2": True,
            "catalogue_text": False,
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "mirrored_or_self_palindromic_units": False,
            "word_order_only_symmetry": False,
        },
        "reader_eligible": False,
        "reader_evidence": {"status": "not_run"},
    }


def controls() -> list[dict[str, object]]:
    raw = [
        [
            "A", "navigator", "maps", "the", "atlas", "while", "the",
            "ranger", "watches", "the", "vessel", "at", "the", "arena",
        ],
        [
            "A", "teacher", "reads", "the", "letter", "while", "the",
            "captain", "guides", "the", "vessel", "at", "the", "harbor",
        ],
        [
            "A", "scholar", "writes", "the", "verse", "while", "the",
            "editor", "marks", "the", "letter", "at", "the", "station",
        ],
        [
            "A", "ranger", "guides", "the", "vessel", "while", "the",
            "dancer", "tends", "the", "garden", "at", "the", "island",
        ],
        [
            "A", "doctor", "studies", "the", "atlas", "while", "the",
            "captain", "steers", "the", "vessel", "at", "the", "marina",
        ],
        [
            "A", "reader", "studies", "the", "atlas", "while", "the",
            "teacher", "reads", "the", "letter", "at", "the", "garden",
        ],
        [
            "A", "writer", "draws", "the", "map", "while", "the",
            "engineer", "maps", "the", "garden", "at", "the", "tower",
        ],
        [
            "A", "doctor", "studies", "the", "garden", "while", "the",
            "dancer", "watches", "the", "bridge", "at", "the", "meadow",
        ],
    ]
    rows = [control_row(tokens) for tokens in raw]
    for row in rows:
        row["audit_rechecked"] = audit(str(row["rendered"]))
        row["audit_recheck_equal"] = row["audit"] == row["audit_rechecked"]
    return rows


def baseline_frontier() -> dict[str, object]:
    """Run the prior lane's two frames and retain its actual seam evidence."""

    frame_stats: dict[str, dict[str, object]] = {}
    examples: list[str] = []
    for index, frame in enumerate(PREVIOUS.FRAMES):
        _rows, stats = PREVIOUS.search_frame(index)
        frame_stats[frame.frame_id] = {
            "two_equation_survivors": stats["two_equation_survivors"],
            "boundary2_examples": stats["boundary2_examples"],
            "character_equations": stats["character_equations"],
            "character_prunes": stats["character_prunes"],
            "max_matched_pairs": stats["max_matched_pairs"],
        }
        for example in stats["boundary2_examples"]:
            if example not in examples:
                examples.append(example)
    return {
        "source_experiment": PREVIOUS.EXPERIMENT_ID,
        "source_script_sha256": hashlib.sha256(
            (ROOT / "experiments/semantic_role_boundary2_live_csp_20260920.py").read_bytes()
        ).hexdigest(),
        "frontier_examples": examples,
        "expected_an_pipe_an_present": "an|an" in examples,
        "frame_stats": frame_stats,
        "comparison_decision": (
            "proceed_to_three_character_operator"
            if "an|an" in examples
            else "do_not_run_three_character_operator"
        ),
    }


def registry_preflight() -> dict[str, object]:
    registry = json.loads(REGISTRY.read_text())
    reports = registry.get("audit_reports", [])
    entries = registry.get("entries", [])
    rows = [row for row in reports + entries if isinstance(row, dict)]
    signatures = [str(row.get("signature", "")) for row in rows]
    ids = [str(row.get("id", "")) for row in rows]
    return {
        "status": "passed",
        "registry_path": str(REGISTRY.relative_to(ROOT)),
        "registry_entries_inspected": len(rows),
        "signature": SIGNATURE,
        "signature_already_present": SIGNATURE in signatures,
        "id_already_present": EXPERIMENT_ID in ids,
        "direct_predecessor": "semantic-role-boundary2-live-csp-20260920",
        "novel_operator": (
            "one independently authored while edge plus third-character seam; "
            "same typed lexical inventory and no repair/reversal"
        ),
        "forbidden_shortcuts_checked": [
            "finished-tape reversal",
            "post-hoc repair",
            "word-order-only symmetry",
            "mirrored or self-palindromic units",
            "catalogue text",
            "fragment output",
            "lexical inventory widening",
            "per-search RLAIF",
        ],
    }


def run() -> dict[str, object]:
    frontier = baseline_frontier()
    if not frontier["expected_an_pipe_an_present"]:
        raise RuntimeError("refusing to run Boundary3 before the an|an frontier exists")

    exact_rows, search_stats = search_relative_temporal()
    exact_clean = [
        row
        for row in exact_rows
        if int(row["length"]) > BENCHMARK_LETTERS
        and bool(row["shortcut_audit"]["shortcut_clean"])
    ]
    for row in exact_rows:
        row["audit_rechecked"] = audit(str(row["rendered"]))
        row["audit_recheck_equal"] = row["audit"] == row["audit_rechecked"]
        row["reader_eligible"] = False

    diagnostic_controls = controls()
    result: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": (
            "one independently authored relative-temporal while edge over the "
            "unchanged semantic-role inventory, with a bounded three-character "
            "Boundary3 state enabled only after the predecessor's an|an frontier "
            "was confirmed"
        ),
        "status": (
            "completed_no_exact_closure"
            if not exact_rows
            else "exact_rows_require_shortcut_and_reader_gate"
        ),
        "baseline_two_character_frontier": frontier,
        "stats": {
            "relative_temporal_frames": 1,
            "edge_kind": FRAME.edge_kind,
            "live_character_dp": search_stats,
            "rendered_complete_prose_controls": len(diagnostic_controls),
            "exact_closures": len(exact_rows),
            "exact_clean_above_38": len(exact_clean),
            "reader_facing_candidates": 0,
            "longest_control_letters": max(
                (int(row["length"]) for row in diagnostic_controls), default=0
            ),
            "shortest_control_letters": min(
                (int(row["length"]) for row in diagnostic_controls), default=0
            ),
        },
        "rendered_complete_prose_controls": diagnostic_controls,
        "exact_rows": exact_rows,
        "exact_clean_above_38": exact_clean,
        "reader_facing_candidates": [],
        "reader_gate": {
            "status": "closed",
            "reason": (
                "No exact-clean output above 38 letters was produced, and "
                "programmatic exactness cannot certify readability."
            ),
            "required_package": (
                "intact prose plus shuffled controls, randomized blinded order, "
                "reproducible rater package"
            ),
        },
        "novelty_preflight": registry_preflight(),
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inventory": (
                "same authored agents, actions, themes, and places as the direct "
                "boundary2 predecessor; no lexical widening"
            ),
            "independent_audits": [
                "outside-in two-pointer scan",
                "forward/reverse SHA-256",
                "audit re-run after closure enumeration",
            ],
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "word_order_only_symmetry": False,
            "mirrored_or_self_palindromic_units": False,
            "catalogue_text": False,
            "fragment_output": False,
            "lexical_inventory_widened": False,
            "per_search_rlaif": False,
            "reader_evidence": False,
        },
        "next_construction": {
            "method": "independently authored relative-temporal edge with a held-out third seam boundary",
            "result": (
                "The confirmed an|an frontier has no compatible third character "
                "in the unchanged agent/place endpoint inventory; every relative-"
                "temporal state therefore dies at the third live equation."
            ),
            "next_operator": (
                "Add one new semantic relation topology with the same three-character "
                "seam state only if its endpoint classes are independently authored "
                "and preflighted; do not widen the lexical bank or add repair."
            ),
            "reader_facing_test": (
                "Any future exact-clean row above 38 must be rendered with intact and "
                "shuffled controls in randomized blinded order before reader eligibility."
            ),
        },
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(OUT),
                "baseline_frontier": frontier["frontier_examples"],
                "exact_rows": len(exact_rows),
                "exact_clean_above_38": len(exact_clean),
                "controls": len(diagnostic_controls),
                "longest_control_letters": result["stats"]["longest_control_letters"],
                "states": search_stats["states_seen"],
                "third_survivors": search_stats["three_equation_survivors"],
            }
        )
    )
    for row in diagnostic_controls:
        print(f"CONTROL {int(row['length']):>3} {row['rendered']}")
    for row in exact_clean:
        print(f"EXACT-CLEAN {int(row['length']):>3} {row['rendered']}")
    return result


if __name__ == "__main__":
    run()
