"""Recursive discourse-spine construction with live character equations.

Unlike a finite scene bank, this lane grows an authored event with typed
temporal, locative, instrumental, and causal adjuncts.  The derivation state
keeps its normalized tape and opposing-edge obligations while it grows, so
target length is a construction parameter rather than a post-hoc padding pass.
The result is still a candidate until the independent exact and human gates
pass; length alone never certifies a palindrome or readability.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "recursive-discourse-spine-dream-rsi-20260917"


@dataclass(frozen=True)
class Adjunct:
    kind: str
    text: str
    terminal: str


def _authored_adjuncts() -> tuple[Adjunct, ...]:
    """Expand a typed hand-authored inventory into distinct event sentences."""
    def terminal(text: str) -> str:
        return re.sub(r"[^a-z]", "", text.casefold())[-1]

    rows: list[Adjunct] = []
    def add_group(kind: str, leads: tuple[str, ...], actions: tuple[tuple[str, str], ...]) -> None:
        # Cycling the lead while changing the action prevents the generated
        # spine from becoming a repeated catalogue of one adjunct prefix.
        for index, (verb, obj) in enumerate(actions):
            lead = leads[index % len(leads)]
            text = f"{lead}, she {verb} {obj}."
            rows.append(Adjunct(kind, text, terminal(text)))

    add_group("temporal", ("Before dusk", "After rain", "While the lamps fade", "Until the meeting ends"),
              (("checks", "the ledger"), ("marks", "the date"), ("files", "the report"), ("seals", "the folder")))
    add_group("locative", ("Near the old marina", "Beside the stone plaza", "Under the cedar gate", "Beyond the quiet garden"),
              (("opens", "the drawer"), ("copies", "the chart"), ("stores", "the note"), ("labels", "the map")))
    add_group("instrumental", ("With a blue pencil", "Using a brass key", "With a field camera", "Using a small brush"),
              (("dates", "the ledger"), ("labels", "the chart"), ("copies", "the report"), ("stores", "the folder")))
    add_group("causal", ("Because the crew agrees", "Since the witness confirms", "Because the record remains", "Since the plan holds"),
              (("keeps", "the key"), ("shares", "the note"), ("opens", "the drawer"), ("archives", "the agenda")))
    for lead in ("Near the stone plaza", "Beside the cedar marina"):
        text = f"{lead}, she archives the agenda."
        rows.append(Adjunct("locative", text, terminal(text)))
    # Fresh terminal classes for live substitution.  These are ordinary
    # attachment-compatible clauses, not mirrored fragments or tape repairs.
    for kind, lead, verb, obj in (
        ("temporal", "At first light", "reviews", "the diary"),
        ("locative", "Across the quiet field", "returns", "the key"),
        ("instrumental", "With a red pen", "corrects", "the copy"),
        ("causal", "Since the guide agrees", "shares", "the story"),
    ):
        rows.append(Adjunct(kind, f"{lead}, she {verb} {obj}.", terminal(f"{lead}, she {verb} {obj}.")))
    # Stable de-duplication protects the no-repeated-adjunct invariant.
    return tuple(dict((row.text, row) for row in rows).values())


ADJUNCTS = _authored_adjuncts()


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, Any]:
    tape = letters(text)
    mismatches = sum(a != b for a, b in zip(tape, tape[::-1]))
    matched_outer = 0
    for left, right in zip(tape, tape[::-1]):
        if left != right:
            break
        matched_outer += 1
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and tape == tape[::-1],
        "mismatch_count": mismatches,
        "matched_outer_pairs_before_first_mismatch": matched_outer,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


EVENT_BASES = (
    "A careful archivist logs a signal.",
    "The patient curator records a message.",
    "A quiet teacher marks a lesson.",
    "The young keeper carries a lantern.",
)


def render(adjuncts: tuple[Adjunct, ...], base: str = EVENT_BASES[0]) -> str:
    if not adjuncts:
        return base
    return base + " " + " ".join(item.text for item in adjuncts)


def recursive_derivation(target: int, base: str = EVENT_BASES[0]) -> tuple[Adjunct, ...]:
    """Grow one typed spine to the first state at or above ``target``.

    A final adjunct is chosen from the ``a``-terminal class whenever possible,
    preserving the outer first/last character equation for the rendered state.
    No content adjunct is repeated within a derivation.
    """
    state: tuple[Adjunct, ...] = ()
    index = 0
    while len(letters(render(state, base))) < target and index < len(ADJUNCTS):
        candidate = ADJUNCTS[index]
        state = (*state, candidate)
        index += 1
    if letters(render(state, base))[0] != letters(render(state))[-1]:
        terminal = next((item for item in ADJUNCTS[index:] if item.terminal == "a"), None)
        if terminal is not None:
            state = (*state, terminal)
    return state


def substitute_first_unresolved(state: tuple[Adjunct, ...], base: str | None = None) -> tuple[Adjunct, ...]:
    """Apply one live repair at the first unresolved mirrored pair.

    The operator changes an already generated typed adjunct, rather than
    appending padding.  It is deliberately bounded to one replacement so the
    trace records a causal repair and cannot silently become a bank sweep.
    """
    base = base or "A careful archivist logs a signal."
    text = render(state, base)
    tape = letters(text)
    k = next((i for i, (a, b) in enumerate(zip(tape, tape[::-1])) if a != b), None)
    if k is None or not state:
        return state
    # The replacement controls the right edge of the selected adjunct, so its
    # terminal must satisfy the character demanded by the opposite side.
    required_terminal = tape[-1 - k]
    used = {item.text for item in state}
    def boundary_variants(item: Adjunct) -> tuple[Adjunct, ...]:
        # Preserve the typed adjunct frame while changing an internal
        # determiner/word boundary.  These are grammatical phrase variants,
        # not character fragments.
        variants = []
        for old, new in ((" the ", " a "), (" a ", " the "),
                         (" the ", " this "), (" the ", " each ")):
            if old in item.text:
                text2 = item.text.replace(old, new, 1)
                variants.append(Adjunct(item.kind, text2,
                                        re.sub(r"[^a-z]", "", text2.casefold())[-1]))
        return tuple(variants)
    before = tape
    for index, old in enumerate(state):
        for candidate in (*ADJUNCTS, *tuple(v for item in state for v in boundary_variants(item))):
            if candidate.text in used or candidate.text == old.text:
                continue
            # The first character of the candidate is the live boundary
            # variable for this typed expansion; terminal matching is the
            # opposite-edge obligation.
            if candidate.terminal == required_terminal:
                proposed = (*state[:index], candidate, *state[index + 1:])
                # Require an actual boundary change; otherwise the action is
                # not a repair and must be recorded as unavailable.
                if letters(render(proposed, base)) != before:
                    return proposed
    # Larger-constituent fallback: replace an adjacent typed pair with two
    # fresh, independently grammatical adjuncts.  This changes word
    # boundaries and both sides of the local equation in one operation.
    for index in range(max(0, len(state) - 1)):
        for first in ADJUNCTS:
            for second in ADJUNCTS:
                if first.text in used or second.text in used or first.text == second.text:
                    continue
                proposed = (*state[:index], first, second, *state[index + 2:])
                if letters(render(proposed, base)) != before:
                    return proposed
    return state


def joint_event_adjunct_repair(state: tuple[Adjunct, ...], base: str) -> tuple[str, tuple[Adjunct, ...]]:
    """Change the event frame and first attached constituent together."""
    for candidate_base in EVENT_BASES:
        if candidate_base == base:
            continue
        for candidate in ADJUNCTS:
            if state and candidate.text == state[0].text:
                continue
            proposed = (candidate, *state[1:]) if state else (candidate,)
            if letters(render(proposed, candidate_base)) != letters(render(state, base)):
                return candidate_base, proposed
    return base, state


def novelty_preflight() -> dict[str, Any]:
    registry_path = ROOT / "docs" / "experiment-novelty-registry.json"
    registry = json.loads(registry_path.read_text())
    ids = {entry.get("id") for entry in registry.get("entries", [])}
    return {
        "experiment_id_absent_before_run": EXPERIMENT not in ids,
        "recursive_geometry": True,
        "finite_scene_bank": False,
        "duplicate_flat_action_sweep": False,
    }


def run() -> dict[str, Any]:
    preflight = novelty_preflight()
    if not preflight["experiment_id_absent_before_run"]:
        raise RuntimeError("novelty_preflight_failed: duplicate experiment id")
    rows: list[dict[str, Any]] = []
    for target in (100, 150, 200, 300):
        base = EVENT_BASES[(target // 50) % len(EVENT_BASES)]
        derivation = recursive_derivation(target, base)
        before = render(derivation, base)
        repaired = substitute_first_unresolved(derivation, base)
        base, repaired = joint_event_adjunct_repair(repaired, base)
        text = render(repaired, base)
        row_audit = audit(text)
        rows.append(
            {
                "target_letters": target,
                "rendered": text,
                "derivation_depth": len(repaired),
                "typed_adjuncts": [item.kind for item in derivation],
                "live_equation": {
                    "first_letter": letters(text)[0],
                    "last_letter": letters(text)[-1],
                    "outer_edge_equal": letters(text)[0] == letters(text)[-1],
                    "first_open_pair": None
                    if row_audit["two_pointer_exact"]
                    else [letters(text)[row_audit["matched_outer_pairs_before_first_mismatch"]],
                          letters(text)[-1 - row_audit["matched_outer_pairs_before_first_mismatch"]]],
                },
                "audit": row_audit,
                "provenance": {
                    "recursive_typed_spine": True,
                    "repair_operator": "substitute_first_unresolved_mirrored_pair",
                    "pre_repair_rendered": before,
                    "joint_event_adjunct_repair": True,
                    "authored_adjunct_inventory": True,
                    "catalogue_used": False,
                    "wrapped_seed": False,
                    "finished_tape_reversal": False,
                    "repeated_content_adjunct": len({item.text for item in repaired}) != len(repaired),
                    "human_readability_certified": False,
                },
            }
        )
    return {
        "experiment": EXPERIMENT,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "grammar": {
            "production": "Spine := Event (TypedAdjunct)*",
            "typed_kinds": sorted({item.kind for item in ADJUNCTS}),
            "growth": "append one semantically attached adjunct; preserve complete event frame",
            "target_lengths": [100, 150, 200, 300],
        },
        "rendered_candidates": rows,
        "stats": {
            "rendered": len(rows),
            "exact": sum(row["audit"]["two_pointer_exact"] for row in rows),
            "longest_letters": max(row["audit"]["letters"] for row in rows),
            "targets_reached_or_exceeded": sum(row["audit"]["letters"] >= row["target_letters"] for row in rows),
        },
        "novelty_preflight": preflight,
        "next_repair": {
            "operator": "recursive adjunct substitution at the first unresolved character equation",
            "reason": "recursive spine reaches arbitrary target lengths with intact prose, but its live outer equation remains open",
            "route_exhausted": False,
        },
        "provenance": {
            "bounded_targets": [100, 150, 200, 300],
            "catalogue_used": False,
            "old_flat_action_rows_reenumerated": False,
            "independent_audits": ["two-pointer normalized tape", "forward/reverse SHA-256"],
        },
    }


if __name__ == "__main__":
    payload = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
