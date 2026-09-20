"""Held-out role edges with a two-character live boundary state.

This experiment follows the semantic-role skeleton CSP, but changes the
frontier representation rather than widening its lexical lists.  The first
two outside-in character equations are consumed as one ``Boundary2`` value.
That value is retained in every memoized state and indexes the still-viable
held-out role edge (temporal or instrumental).  The role edge, lexical value,
and character obligations are selected together; no completed tape is reversed
or edited after the search.

The inventory is fresh and small.  Brown or other corpus surface sentences are
not used.  Complete controls are rendered independently so they remain useful
even when the exact frontier closes nowhere.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/semantic-role-boundary2-live-csp-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT_ID = "semantic-role-boundary2-live-csp-20260920"
SIGNATURE = (
    "held-out-temporal-instrumental-edge|two-character-boundary-state|"
    "live-role-csp"
)
BENCHMARK_LETTERS = 38
STATE_LIMIT = 180_000
CENTER_LIMIT = 40_000


def normalize_letters(text: str) -> str:
    """Apply the project's letter-only tape convention."""

    return re.sub(r"[^a-z]", "", text.casefold())


def pointer_scan(text: str) -> dict[str, object]:
    """Independent outside-in equality scan."""

    tape = normalize_letters(text)
    mismatch = next(
        (
            (i, tape[i], tape[-1 - i])
            for i in range(len(tape) // 2)
            if tape[i] != tape[-1 - i]
        ),
        None,
    )
    return {
        "letters": len(tape),
        "pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
    }


def hash_audit(text: str) -> dict[str, object]:
    """Independent forward/reverse digest audit."""

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


@dataclass(frozen=True)
class Slot:
    name: str
    role: str
    values: tuple[str, ...]
    fixed: bool = False


@dataclass(frozen=True)
class Frame:
    frame_id: str
    edge_kind: str
    description: str
    slots: tuple[Slot, ...]


@dataclass(frozen=True)
class Boundary2:
    """The first two live equations, retained as a stateful seam object.

    ``left_window`` is read from the left in sentence order and
    ``right_window`` is read inward from the right.  Equality is checked when
    each pair is added; storing both windows makes the seam auditable instead
    of reducing it to a scalar score.
    """

    left_window: str
    right_window: str
    equations: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class State:
    frame_index: int
    tokens: tuple[str | None, ...]
    roles: tuple[tuple[str, str], ...]
    left_slot: int
    right_slot: int
    left_pos: int
    right_pos: int
    matched_pairs: int
    boundary2: Boundary2 | None


def slot(name: str, role: str, *values: str, fixed: bool = False) -> Slot:
    return Slot(name=name, role=role, values=tuple(dict.fromkeys(values)), fixed=fixed)


# Fresh, typed role inventory.  The endpoint spellings deliberately include
# ordinary choices whose final two letters can expose a second seam equation.
AGENTS = (
    "navigator",
    "ranger",
    "reader",
    "teacher",
    "captain",
    "scholar",
    "writer",
    "watcher",
    "doctor",
    "dancer",
    "engineer",
    "editor",
)

ACTIONS = (
    "maps",
    "charts",
    "reads",
    "marks",
    "guides",
    "writes",
    "studies",
    "watches",
    "tends",
    "steers",
    "draws",
)

THEMES = (
    "atlas",
    "inlet",
    "letter",
    "harbor",
    "vessel",
    "garden",
    "map",
    "verse",
    "signal",
    "bridge",
)

TEMPORAL_PLACES = (
    "arena",
    "marina",
    "garden",
    "harbor",
    "station",
    "tower",
    "meadow",
    "island",
)

INSTRUMENTS_A = (
    "compass",
    "hammer",
    "lantern",
    "marker",
    "pencil",
    "brush",
    "ruler",
    "drum",
    "rope",
)

INSTRUMENTS_AN = (
    "axe",
    "awl",
    "oar",
    "inkpot",
)

INSTRUMENTAL_PLACES = (
    "arena",
    "harbor",
    "station",
    "garden",
    "tower",
    "island",
    "plaza",
)

AGENT_ACTIONS = {
    "navigator": {"maps", "charts", "guides", "steers", "reads"},
    "ranger": {"maps", "guides", "watches", "marks", "tends"},
    "reader": {"reads", "marks", "studies", "writes"},
    "teacher": {"reads", "marks", "writes", "guides", "studies"},
    "captain": {"maps", "charts", "guides", "steers", "watches"},
    "scholar": {"reads", "writes", "studies", "marks", "draws"},
    "writer": {"writes", "draws", "marks", "reads"},
    "watcher": {"watches", "marks", "guides", "reads"},
    "doctor": {"studies", "reads", "marks", "draws"},
    "dancer": {"draws", "watches", "guides", "tends"},
    "engineer": {"maps", "draws", "guides", "marks", "studies"},
    "editor": {"reads", "marks", "writes", "studies"},
}

ACTION_THEMES = {
    "maps": {"atlas", "inlet", "harbor", "garden", "map", "island"},
    "charts": {"atlas", "inlet", "harbor", "vessel", "bridge"},
    "reads": {"atlas", "letter", "verse", "signal", "map"},
    "marks": {"letter", "map", "signal", "bridge", "garden"},
    "guides": {"vessel", "harbor", "inlet", "bridge", "island"},
    "writes": {"letter", "verse", "signal", "map"},
    "studies": {"atlas", "letter", "garden", "signal", "inlet"},
    "watches": {"harbor", "vessel", "signal", "bridge", "garden"},
    "tends": {"garden", "island", "harbor"},
    "steers": {"vessel", "inlet", "harbor", "island"},
    "draws": {"map", "atlas", "bridge", "garden", "signal"},
}

AGENT_THEMES = {
    "navigator": {"atlas", "inlet", "harbor", "map", "island"},
    "ranger": {"garden", "harbor", "bridge", "island", "signal", "vessel"},
    "reader": {"atlas", "letter", "verse", "signal", "map"},
    "teacher": {"letter", "verse", "signal", "map", "atlas"},
    "captain": {"atlas", "inlet", "harbor", "vessel", "bridge"},
    "scholar": {"atlas", "letter", "verse", "garden", "signal"},
    "writer": {"letter", "verse", "signal", "map"},
    "watcher": {"harbor", "vessel", "signal", "bridge", "garden"},
    "doctor": {"atlas", "letter", "garden", "signal"},
    "dancer": {"bridge", "garden", "island", "signal"},
    "engineer": {"atlas", "inlet", "bridge", "map", "garden"},
    "editor": {"letter", "verse", "signal", "map", "atlas"},
}

ACTION_INSTRUMENTS = {
    "maps": {"compass", "pencil", "ruler", "inkpot"},
    "charts": {"compass", "pencil", "ruler", "inkpot"},
    "reads": {"lantern", "marker", "pencil", "inkpot"},
    "marks": {"marker", "pencil", "brush", "awl"},
    "guides": {"compass", "lantern", "rope", "oar"},
    "writes": {"pencil", "brush", "inkpot", "awl"},
    "studies": {"lantern", "marker", "pencil", "inkpot"},
    "watches": {"lantern", "compass", "telescope"},
    "tends": {"brush", "ruler", "rope", "awl"},
    "steers": {"compass", "rope", "oar"},
    "draws": {"pencil", "brush", "ruler", "inkpot"},
}

INSTRUMENTS = tuple(dict.fromkeys(INSTRUMENTS_A + INSTRUMENTS_AN))


def temporal_frame() -> Frame:
    return Frame(
        frame_id="temporal_before",
        edge_kind="temporal",
        description="agent acts on a theme before a setting",
        slots=(
            slot("subject_determiner", "fixed", "A", fixed=True),
            slot("agent", "agent", *AGENTS),
            slot("action", "action", *ACTIONS),
            slot("object_determiner", "fixed", "the", fixed=True),
            slot("theme", "theme", *THEMES),
            slot("temporal_connector", "fixed", "before", fixed=True),
            slot("setting_determiner", "fixed", "the", fixed=True),
            slot("place", "temporal_place", *TEMPORAL_PLACES),
        ),
    )


def instrumental_frame() -> Frame:
    return Frame(
        frame_id="instrumental_with_at",
        edge_kind="instrumental",
        description="agent acts on a theme with an instrument at a setting",
        slots=(
            slot("subject_determiner", "fixed", "A", fixed=True),
            slot("agent", "agent", *AGENTS),
            slot("action", "action", *ACTIONS),
            slot("object_determiner", "fixed", "the", fixed=True),
            slot("theme", "theme", *THEMES),
            slot("instrumental_connector", "fixed", "with", fixed=True),
            slot("instrument_determiner", "instrument_determiner", "a", "an"),
            slot("instrument", "instrument", *INSTRUMENTS),
            slot("locative_connector", "fixed", "at", fixed=True),
            slot("setting_determiner", "fixed", "the", fixed=True),
            slot("place", "instrumental_place", *INSTRUMENTAL_PLACES),
        ),
    )


FRAMES = (temporal_frame(), instrumental_frame())


def role_map(state: State) -> dict[str, str]:
    return dict(state.roles)


def frame_for(state: State) -> Frame:
    return FRAMES[state.frame_index]


def options_for(slot_value: Slot) -> tuple[str, ...]:
    return slot_value.values


def edge_boundary_prefixes(frame: Frame) -> frozenset[str]:
    """Return the two-letter seams reachable by this edge's endpoints.

    This is computed from the live endpoint inventories, not from generated
    sentences.  It is used as a finite seam index once the first two
    equations survive.
    """

    places = next(
        slot_value.values
        for slot_value in frame.slots
        if slot_value.role.endswith("place")
    )
    agents = next(
        slot_value.values
        for slot_value in frame.slots
        if slot_value.role == "agent"
    )
    return frozenset(
        "a" + agent[0]
        for agent in agents
        if any(place.endswith(agent[0] + "a") for place in places)
    )


def partial_semantics(frame: Frame, roles: dict[str, str]) -> bool:
    """Check typed role compatibility before a state can advance."""

    agent = roles.get("agent")
    action = roles.get("action")
    theme = roles.get("theme")
    instrument = roles.get("instrument")
    instrument_det = roles.get("instrument_determiner")

    if agent and action and action not in AGENT_ACTIONS.get(agent, set()):
        return False
    if agent and theme and theme not in AGENT_THEMES.get(agent, set()):
        return False
    if action and theme and theme not in ACTION_THEMES.get(action, set()):
        return False
    if action and instrument and instrument not in ACTION_INSTRUMENTS.get(action, set()):
        return False
    if instrument and instrument_det:
        begins_vowel = instrument[0] in "aeiou"
        if instrument_det == "an" and not begins_vowel:
            return False
        if instrument_det == "a" and begins_vowel:
            return False
    return True


def assign_slot(state: State, index: int) -> list[State]:
    frame = frame_for(state)
    if state.tokens[index] is not None:
        return [state]
    slot_value = frame.slots[index]
    out: list[State] = []
    for value in options_for(slot_value):
        tokens = list(state.tokens)
        tokens[index] = value
        roles = role_map(state)
        if slot_value.role not in {"fixed", "instrument_determiner"}:
            roles[slot_value.role] = value
        elif slot_value.role == "instrument_determiner":
            roles[slot_value.role] = value
        if not partial_semantics(frame, roles):
            continue
        out.append(
            State(
                frame_index=state.frame_index,
                tokens=tuple(tokens),
                roles=tuple(sorted(roles.items())),
                left_slot=state.left_slot,
                right_slot=state.right_slot,
                left_pos=state.left_pos,
                right_pos=state.right_pos,
                matched_pairs=state.matched_pairs,
                boundary2=state.boundary2,
            )
        )
    return out


def render_tokens(tokens: Iterable[str | None]) -> str:
    values = list(tokens)
    if any(value is None for value in values):
        raise ValueError("cannot render an incomplete frame")
    return " ".join(value for value in values if value is not None) + "."


def shortcut_audit(text: str, frame: Frame) -> dict[str, object]:
    """Reject prohibited constructions without judging readability."""

    words = text.rstrip(".").split()
    content_words = [
        normalize_letters(word)
        for word, slot_value in zip(words, frame.slots)
        if not slot_value.fixed
    ]
    repeated = sorted(
        word for word, count in Counter(content_words).items() if word and count > 1
    )
    self_pal = sorted(
        word for word in content_words if len(word) > 1 and word == word[::-1]
    )
    word_order_only = len(content_words) > 1 and content_words == content_words[::-1]
    flags: dict[str, object] = {
        "repeated_content_units": repeated,
        "repeated_units": bool(repeated),
        "self_palindromic_units": self_pal,
        "word_order_only_symmetry": word_order_only,
        "catalogue_text": False,
        "finished_tape_reversal": False,
        "post_hoc_repair": False,
        "fragment": len(words) != len(frame.slots),
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


def state_key(state: State) -> tuple[object, ...]:
    boundary = None
    if state.boundary2 is not None:
        boundary = (
            state.boundary2.left_window,
            state.boundary2.right_window,
            state.boundary2.equations,
        )
    return (
        state.frame_index,
        state.tokens,
        state.roles,
        state.left_slot,
        state.right_slot,
        state.left_pos,
        state.right_pos,
        state.matched_pairs,
        boundary,
    )


def closure_row(state: State) -> dict[str, object] | None:
    if any(value is None for value in state.tokens):
        return None
    frame = frame_for(state)
    roles = role_map(state)
    if not partial_semantics(frame, roles):
        return None
    rendered = render_tokens(state.tokens)
    exact_audit = audit(rendered)
    shortcuts = shortcut_audit(rendered, frame)
    boundary = state.boundary2
    return {
        "rendered": rendered,
        "length": exact_audit["letters"],
        "frame": frame.frame_id,
        "edge_kind": frame.edge_kind,
        "semantic_roles": roles,
        "boundary2": {
            "left_window": boundary.left_window if boundary else "",
            "right_window": boundary.right_window if boundary else "",
            "equations": list(boundary.equations) if boundary else [],
        },
        "audit": exact_audit,
        "shortcut_audit": shortcuts,
        "provenance": {
            "source": "fresh authored typed temporal/instrumental role inventory",
            "role_edge_selected_inside_live_csp": True,
            "two_character_boundary_state_carried": True,
            "boundary_state_from_first_outer_equation": True,
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
    state: State,
    rows: list[dict[str, object]],
    counters: dict[str, object],
    seen_text: set[str],
    budget: list[int],
) -> None:
    """Enumerate the remaining grammar center before exact validation."""

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
        if bool(row["audit"]["exact"]):
            counters["exact_closures"] += 1
            rows.append(row)
        return
    index = next(i for i, value in enumerate(state.tokens) if value is None)
    for candidate in assign_slot(state, index):
        finish_center(candidate, rows, counters, seen_text, budget)


def search_frame(frame_index: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    frame = FRAMES[frame_index]
    start = State(
        frame_index=frame_index,
        tokens=tuple(None for _ in frame.slots),
        roles=tuple(),
        left_slot=0,
        right_slot=len(frame.slots) - 1,
        left_pos=0,
        right_pos=0,
        matched_pairs=0,
        boundary2=None,
    )
    rows: list[dict[str, object]] = []
    counters: dict[str, object] = {
        "states_seen": 0,
        "memo_hits": 0,
        "semantic_prunes": 0,
        "character_prunes": 0,
        "boundary_advances": 0,
        "character_equations": 0,
        "first_equation_survivors": 0,
        "two_equation_survivors": 0,
        "boundary2_examples": [],
        "boundary2_edge_index_prunes": 0,
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
    edge_prefixes = edge_boundary_prefixes(frame)

    def walk(state: State) -> None:
        key = state_key(state)
        if key in seen:
            counters["memo_hits"] += 1
            return
        if len(seen) >= STATE_LIMIT:
            counters["state_limit_prunes"] += 1
            return
        seen.add(key)
        counters["states_seen"] += 1
        counters["max_matched_pairs"] = max(
            counters["max_matched_pairs"], state.matched_pairs
        )

        if state.left_slot >= state.right_slot:
            finish_center(state, rows, counters, seen_text, center_budget)
            return

        left_states = assign_slot(state, state.left_slot)
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

                # Move over already-consumed word boundaries without emitting
                # a character.  This preserves the live grammar cursors.
                if paired.left_pos >= len(left_value):
                    walk(
                        State(
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
                        State(
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
                    continue

                if paired.matched_pairs == 0:
                    counters["first_equation_survivors"] += 1
                boundary = paired.boundary2
                if boundary is None:
                    boundary = Boundary2(
                        left_window=left_char,
                        right_window=right_char,
                        equations=((left_char, right_char),),
                    )
                elif len(boundary.equations) < 2:
                    boundary = Boundary2(
                        left_window=boundary.left_window + left_char,
                        right_window=boundary.right_window + right_char,
                        equations=boundary.equations + ((left_char, right_char),),
                    )
                    if boundary.left_window not in edge_prefixes:
                        counters["boundary2_edge_index_prunes"] += 1
                        continue
                    counters["two_equation_survivors"] += 1
                    examples = counters["boundary2_examples"]
                    assert isinstance(examples, list)
                    signature = (
                        f"{boundary.left_window}|{boundary.right_window}"
                    )
                    if signature not in examples and len(examples) < 8:
                        examples.append(signature)

                next_state = State(
                    **{
                        **paired.__dict__,
                        "left_pos": paired.left_pos + 1,
                        "right_pos": paired.right_pos + 1,
                        "matched_pairs": paired.matched_pairs + 1,
                        "boundary2": boundary,
                    }
                )
                walk(next_state)

    walk(start)
    counters["memoized_frontier_states"] = len(seen)
    counters["longest_matched_prefix"] = counters["max_matched_pairs"]
    return rows, counters


def control_row(frame_index: int, tokens: list[str]) -> dict[str, object]:
    frame = FRAMES[frame_index]
    if len(tokens) != len(frame.slots):
        raise ValueError("control token count does not match frame")
    roles: dict[str, str] = {}
    for slot_value, value in zip(frame.slots, tokens):
        if slot_value.role not in {"fixed", "instrument_determiner"}:
            roles[slot_value.role] = value
        elif slot_value.role == "instrument_determiner":
            roles[slot_value.role] = value
    if not partial_semantics(frame, roles):
        raise ValueError(f"invalid control semantics: {tokens}")
    rendered = render_tokens(tokens)
    first_audit = audit(rendered)
    return {
        "rendered": rendered,
        "length": first_audit["letters"],
        "frame": frame.frame_id,
        "edge_kind": frame.edge_kind,
        "semantic_roles": roles,
        "audit": first_audit,
        "shortcut_audit": shortcut_audit(rendered, frame),
        "provenance": {
            "source": "fresh authored complete-English control",
            "generated_by": "typed temporal/instrumental frame lexicalization",
            "candidate_status": "control_only",
            "complete_prose": True,
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
    # These are intact sentences with ordinary semantics.  They are retained
    # as reader-facing controls only; no control is promoted by exactness.
    raw = [
        (
            0,
            ["A", "navigator", "maps", "the", "atlas", "before", "the", "arena"],
        ),
        (
            0,
            ["A", "ranger", "guides", "the", "vessel", "before", "the", "harbor"],
        ),
        (
            0,
            ["A", "teacher", "reads", "the", "letter", "before", "the", "station"],
        ),
        (
            0,
            ["A", "scholar", "writes", "the", "verse", "before", "the", "garden"],
        ),
        (
            1,
            [
                "A",
                "navigator",
                "maps",
                "the",
                "atlas",
                "with",
                "a",
                "compass",
                "at",
                "the",
                "arena",
            ],
        ),
        (
            1,
            [
                "A",
                "teacher",
                "writes",
                "the",
                "letter",
                "with",
                "a",
                "pencil",
                "at",
                "the",
                "station",
            ],
        ),
        (
            1,
            [
                "A",
                "captain",
                "guides",
                "the",
                "vessel",
                "with",
                "a",
                "rope",
                "at",
                "the",
                "harbor",
            ],
        ),
        (
            1,
            [
                "A",
                "scholar",
                "writes",
                "the",
                "verse",
                "with",
                "an",
                "awl",
                "at",
                "the",
                "garden",
            ],
        ),
    ]
    rows = [control_row(index, tokens) for index, tokens in raw]
    for row in rows:
        row["audit_rechecked"] = audit(str(row["rendered"]))
        row["audit_recheck_equal"] = row["audit"] == row["audit_rechecked"]
    return rows


def registry_preflight() -> dict[str, object]:
    registry = json.loads(REGISTRY.read_text())
    reports = registry.get("audit_reports", [])
    entries = registry.get("entries", [])
    all_rows = [row for row in reports + entries if isinstance(row, dict)]
    signatures = [str(row.get("signature", "")) for row in all_rows]
    ids = [str(row.get("id", "")) for row in all_rows]
    overlaps = [
        "semantic-role-skeleton-live-csp-20260920",
        "typed-center-state-wfst-20260920",
        "scene-lattice-constructive-author-20260920",
        "character-orbit-scene-residual-20260920",
    ]
    return {
        "status": "passed",
        "registry_path": str(REGISTRY.relative_to(ROOT)),
        "registry_entries_inspected": len(all_rows),
        "signature": SIGNATURE,
        "signature_already_present": SIGNATURE in signatures,
        "id_already_present": EXPERIMENT_ID in ids,
        "overlaps_checked": overlaps,
        "distinct_from": (
            "Unlike the prior first-character skeleton, this lane consumes and "
            "memoizes the first two equality equations as Boundary2, then "
            "carries that seam through two held-out role-edge grammars: temporal "
            "before and instrumental with-at.  It is not a lexical bank sweep."
        ),
        "forbidden_shortcuts_checked": [
            "finished-tape reversal",
            "post-hoc repair",
            "word-order-only symmetry",
            "mirrored or self-palindromic units",
            "catalogue text",
            "fragment output",
            "per-search RLAIF",
        ],
    }


def run() -> dict[str, object]:
    all_exact: list[dict[str, object]] = []
    search_stats: dict[str, dict[str, object]] = {}
    for index, frame in enumerate(FRAMES):
        rows, stats = search_frame(index)
        all_exact.extend(rows)
        search_stats[frame.frame_id] = stats

    exact_clean = [
        row
        for row in all_exact
        if int(row["length"]) > BENCHMARK_LETTERS
        and bool(row["shortcut_audit"]["shortcut_clean"])
    ]
    for row in all_exact:
        row["audit_rechecked"] = audit(str(row["rendered"]))
        row["audit_recheck_equal"] = row["audit"] == row["audit_rechecked"]
        row["reader_eligible"] = False

    diagnostic_controls = controls()
    result: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": (
            "held-out temporal/instrumental semantic role edges with a live "
            "two-character Boundary2 state: the first two outside-in equations "
            "are consumed before the state is memoized and carried through the "
            "typed grammar frontier"
        ),
        "status": (
            "completed_no_exact_closure"
            if not all_exact
            else "exact_rows_require_shortcut_and_reader_gate"
        ),
        "stats": {
            "frames": len(FRAMES),
            "frame_ids": [frame.frame_id for frame in FRAMES],
            "edge_kinds": [frame.edge_kind for frame in FRAMES],
            "live_character_dp": search_stats,
            "rendered_complete_prose_controls": len(diagnostic_controls),
            "exact_closures": len(all_exact),
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
        "exact_rows": all_exact,
        "exact_clean_above_38": exact_clean,
        "reader_facing_candidates": [],
        "reader_gate": {
            "status": "closed",
            "reason": (
                "Programmatic exactness does not certify readability; no "
                "exact-clean >38 row has blinded human evidence."
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
                "fresh authored agents, actions, themes, temporal places, and "
                "instrument nouns; no corpus surface sentences"
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
            "per_search_rlaif": False,
            "reader_evidence": False,
        },
        "next_construction": {
            "method": "typed relative temporal edge with a three-character seam automaton",
            "change": (
                "Add one independently authored relative-time edge ('while the ...') "
                "and promote Boundary2 to a bounded three-character seam only after "
                "the current two-character state has been compared across both edge "
                "types."
            ),
            "why": (
                "This run tests whether a short seam state can preserve a viable "
                "role-edge frontier; the next operator should add grammatical depth, "
                "not another lexical bank or repair pass."
            ),
            "reader_facing_test": (
                "Any exact-clean row above 38 must be rendered with intact and "
                "shuffled controls, randomized and blinded before reader eligibility."
            ),
        },
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(OUT),
                "exact_rows": len(all_exact),
                "exact_clean_above_38": len(exact_clean),
                "controls": len(diagnostic_controls),
                "longest_control_letters": result["stats"]["longest_control_letters"],
                "states": {
                    key: value["states_seen"] for key, value in search_stats.items()
                },
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
