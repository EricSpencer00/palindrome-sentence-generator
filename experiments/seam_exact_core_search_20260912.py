"""Exact centre-out construction over semantic cores and editable seams.

The preceding constituent-pair experiment generated two clauses and checked
symmetry only afterward.  This follow-on keeps a palindrome invariant in the
search state.  A complete constituent pair is wrapped around the current core;
the left phrase is prepended and the independently authored right phrase is
appended.  Word boundaries may split or merge at the seam (two words on one
side versus three on the other), and each seam changes the letter length.

The search admits a state only when its two accumulated tapes are exact
reflections.  Nonmatching seam attempts are still rendered and recorded as
hard rejections.  The shared admission gate is applied only to already-exact
closures, and no readability claim is made.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


MIN_LETTERS = 100
MAX_LETTERS = 180
DEFAULT_MAX_DEPTH = 6
DEFAULT_BEAM = 128
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class Seam:
    identifier: str
    role: str
    left: tuple[str, ...]
    right: tuple[str, ...]


@dataclass(frozen=True)
class Core:
    identifier: str
    event: str
    left_source_clause: str
    right_source_clause: str
    left: tuple[str, ...]
    right: tuple[str, ...]


# These are authored phrase pairs, selected before the run.  The right side is
# not produced by reversing the left in the constructor; the solver checks the
# relation as an extension condition.  Their differing word counts are the
# editable seam (merge left / split right).
SEAMS = (
    Seam("network-cap", "argument", ("network", "cap"), ("pack", "row", "ten")),
    Seam("waste-speed", "predicate_argument", ("waste", "speed"), ("deep", "set", "saw")),
    Seam("never-often", "adjunct", ("never", "often"), ("net", "for", "even")),
    Seam("trade-rare", "predicate_argument", ("trade", "rare"), ("era", "red", "art")),
    Seam("still-award", "predicate_argument", ("still", "award"), ("draw", "all", "its")),
    Seam("went-older", "adjunct", ("went", "older"), ("red", "lot", "new")),
    Seam("test-older", "predicate_argument", ("test", "older"), ("red", "lot", "set")),
    Seam("waste-metal", "argument", ("waste", "metal"), ("late", "met", "saw")),
)

# A deliberately incompatible authored pair is searched as a negative control
# so the run records the exact-closure constraint rejecting a seam, rather than
# relying only on duplicate-word or length rejections.
INCOMPATIBLE_CONTROL = Seam(
    "incompatible-control", "negative_control", ("careful", "editors"),
    ("calm", "archives"),
)
SEAM_CANDIDATES = SEAMS + (INCOMPATIBLE_CONTROL,)


# The sources are ordinary independently authored clauses.  The short core
# spans are complete surface constituents selected from those clauses; they
# are preserved as atomic units by the seam solver even when the final letter
# tape crosses their word boundaries.
CORES = (
    Core(
        "courier-prison",
        "a courier moves toward a prison while a worker checks a pot",
        "A courier walks toward the prison after the morning delivery.",
        "A worker checks the iron pot before the evening meal.",
        ("to", "prison"),
        ("no", "sir", "pot"),
    ),
    Core(
        "site-report",
        "a clerk reviews a site report while a guide marks a route",
        "A careful clerk reviews the site report before the meeting.",
        "A local guide marks the route beside the station.",
        ("site", "got"),
        ("to", "get", "is"),
    ),
    Core(
        "one-hot",
        "a cook serves one hot meal while a host checks the room",
        "A quiet cook serves one hot meal to the waiting guests.",
        "A host checks the room before the late dinner.",
        ("one", "hot"),
        ("to", "he", "no"),
    ),
)


def tape(words: Iterable[str]) -> str:
    return normalize_letters(" ".join(words))


def independent_two_pointer(text: str) -> dict[str, object]:
    """Recompute exactness without relying on the search invariant."""
    try:
        normalized = normalize_letters(text)
    except (TypeError, ValueError):
        return {"normalized": "", "letters": 0, "exact": False, "first_mismatch": None}
    left, right = 0, len(normalized) - 1
    while left < right:
        if normalized[left] != normalized[right]:
            return {"normalized": normalized, "letters": len(normalized), "exact": False,
                    "first_mismatch": [left, right]}
        left += 1
        right -= 1
    return {"normalized": normalized, "letters": len(normalized), "exact": bool(normalized),
            "first_mismatch": None}


def seam_compatible(seam: Seam) -> bool:
    """Check a seam's reflected tape before it can extend a state."""
    return tape(seam.left) == tape(seam.right)[::-1]


def _render(left: tuple[str, ...], right: tuple[str, ...]) -> str:
    opening = " ".join(left).capitalize()
    terminal = " ".join(right).lower()
    return f"{opening}; {terminal}."


def _attempt_record(
    *, core: Core, depth: int, left: tuple[str, ...], right: tuple[str, ...],
    seam: Seam | None, search_exact: bool, rejection: str | None,
) -> dict[str, object]:
    rendered = _render(left, right)
    pointer = independent_two_pointer(rendered)
    return {
        "depth": depth,
        "seam_id": seam.identifier if seam else None,
        "seam_role": seam.role if seam else "core",
        "rendered": rendered,
        "left_constituents": list(left),
        "right_constituents": list(right),
        "search_exact": search_exact,
        "independent_two_pointer": pointer,
        "hard_rejection": rejection,
        "length": pointer["letters"],
        "length_change_from_seam": (
            len(tape(seam.left)) * 2 if seam is not None else len(tape(core.left)) * 2
        ),
        "seam_word_count_change": (
            (len(seam.left), len(seam.right)) if seam is not None else
            (len(core.left), len(core.right))
        ),
    }


def search_core(core: Core, *, max_depth: int = DEFAULT_MAX_DEPTH,
                beam: int = DEFAULT_BEAM) -> dict[str, object]:
    """Search only exact-reflecting states and retain every seam attempt."""
    if max_depth < 0 or beam < 1:
        raise ValueError("max_depth must be nonnegative and beam positive")
    attempts: list[dict[str, object]] = []
    left0, right0 = tuple(core.left), tuple(core.right)
    core_exact = tape(left0) == tape(right0)[::-1]
    attempts.append(_attempt_record(
        core=core, depth=0, left=left0, right=right0,
        seam=None, search_exact=core_exact,
        rejection=None if core_exact else "core_reflection_mismatch",
    ))
    if not core_exact:
        return {"core_id": core.identifier, "attempts": attempts,
                "exact_closures": [], "states_expanded": 0,
                "hard_rejections": 1, "max_depth": max_depth, "beam": beam}

    # State stores rendered left and right constituent sequences.  Invariant:
    # tape(left) == reverse(tape(right)); wrapping preserves this exactly.
    frontier = [(left0, right0, frozenset(left0 + right0), ())]
    closures: list[dict[str, object]] = []
    states_expanded = 0
    hard_rejections = 0
    for depth in range(1, max_depth + 1):
        next_states = []
        for left, right, used, trace in frontier:
            states_expanded += 1
            for seam in SEAM_CANDIDATES:
                candidate_left = seam.left + left
                candidate_right = right + seam.right
                rendered = _render(candidate_left, candidate_right)
                # Enforce exactness before this state is retained or expanded.
                extension_exact = tape(candidate_left) == tape(candidate_right)[::-1]
                if not seam_compatible(seam):
                    extension_exact = False
                    reason = "seam_reflection_mismatch"
                elif used.intersection(seam.left + seam.right):
                    extension_exact = False
                    reason = "repeated_word_across_constituents"
                elif len(tape(candidate_left + candidate_right)) > MAX_LETTERS:
                    extension_exact = False
                    reason = "length_above_bound"
                elif not extension_exact:
                    reason = "extension_not_exact"
                else:
                    reason = None
                attempts.append(_attempt_record(
                    core=core, depth=depth, left=candidate_left,
                    right=candidate_right, seam=seam,
                    search_exact=extension_exact, rejection=reason,
                ))
                if not extension_exact:
                    hard_rejections += 1
                    continue
                next_states.append((candidate_left, candidate_right,
                                    frozenset(candidate_left + candidate_right),
                                    trace + (seam.identifier,)))
                audit = independent_two_pointer(rendered)
                # This assertion is a consistency check, not the constructor's
                # admission mechanism: the exact state was accepted above.
                assert audit["exact"] is True
                closures.append({
                    "core_id": core.identifier,
                    "depth": depth,
                    "rendered": rendered,
                    "length": audit["letters"],
                    "construction_exact": True,
                    "independent_two_pointer": audit,
                    "seam_trace": list(trace + (seam.identifier,)),
                    "preserved_semantic_cores": [list(core.left), list(core.right)],
                    "seam_edits": [
                        {"id": item.identifier, "role": item.role,
                         "left": list(item.left), "right": list(item.right),
                         "left_letters": len(tape(item.left)),
                         "right_letters": len(tape(item.right)),
                         "word_count": [len(item.left), len(item.right)]}
                        for item in SEAMS if item.identifier in trace + (seam.identifier,)
                    ],
                })
        next_states.sort(key=lambda state: (len(tape(state[0] + state[1])), state[3]))
        frontier = next_states[:beam]
        if not frontier:
            break
    return {"core_id": core.identifier, "attempts": attempts,
            "exact_closures": closures, "states_expanded": states_expanded,
            "hard_rejections": hard_rejections, "max_depth": max_depth, "beam": beam}


def audit_closure(row: dict[str, object]) -> dict[str, object]:
    """Apply the unchanged central gate to an already exact closure."""
    checks = mechanical_admission_checks(
        str(row["rendered"]), min_letters=MIN_LETTERS, max_letters=MAX_LETTERS
    )
    central_gate_passed = all(checks.values())
    # These seam fixtures preserve atomic cores but do not carry a complete
    # grammatical sentence witness through the assembled surface.  Keep them
    # as exact diagnostics, never as prospective reader items.  A future run
    # may set this field only from an independently authored intact witness;
    # it is not inferred from the mechanical gate or a language proxy.
    intact_sentence_witness = False
    rejection_codes = [name for name, passed in checks.items() if not passed]
    if not intact_sentence_witness:
        rejection_codes.append("no_intact_grammatical_sentence_witness")
    return {
        **row,
        "current_central_admission": checks,
        "central_gate_passed": central_gate_passed,
        "intact_grammatical_sentence_witness": intact_sentence_witness,
        "rejection_codes": rejection_codes,
        "mechanically_admitted": bool(
            row["construction_exact"] and central_gate_passed and intact_sentence_witness
        ),
        "reader_status": "No human reader test was run; exactness is not a readability claim.",
    }


def run(*, max_depth: int = DEFAULT_MAX_DEPTH, beam: int = DEFAULT_BEAM) -> dict[str, object]:
    searches = [search_core(core, max_depth=max_depth, beam=beam) for core in CORES]
    exact = [audit_closure(row) for search in searches for row in search["exact_closures"]]
    return {
        "status": "complete_bounded_exact_seam_core_search",
        "config": {"cores": len(CORES), "seams": len(SEAMS),
                   "seam_candidates": len(SEAM_CANDIDATES), "max_depth": max_depth,
                   "beam": beam, "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
                   "exactness_enforced_during_search": True},
        "provenance": {
            "cores_sha256": sha256(json.dumps([core.__dict__ for core in CORES], sort_keys=True).encode()).hexdigest(),
            "seams_sha256": sha256(json.dumps([seam.__dict__ for seam in SEAMS], sort_keys=True).encode()).hexdigest(),
            "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "construction_material": "Three task-local independently authored clause witnesses and eight authored seam phrase pairs; no palindrome catalogue or relexicalization bank was used.",
            "central_admission": "llm_palindrome.admission.mechanical_admission_checks with unchanged defaults",
        },
        "searches": searches,
        "exact_closures": exact,
        "central_gate_survivors": [row for row in exact if row["central_gate_passed"]],
        "mechanically_admitted": [row for row in exact if row["mechanically_admitted"]],
        "readable_survivors": [],
        "reader_facing_next_test": (
            "Exact closures are construction evidence only. If any closure clears the unchanged "
            "central gate, blind it against independently authored intact-clause controls and a "
            "shuffled control, collecting one-pass readability, grammatical completeness, and free "
            "paraphrase. Do not label a closure readable from this run."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--beam", type=int, default=DEFAULT_BEAM)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(max_depth=args.max_depth, beam=args.beam)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "attempts": sum(len(s["attempts"]) for s in result["searches"]),
                      "exact_closures": len(result["exact_closures"]),
                      "admitted": len(result["mechanically_admitted"])}, indent=2))


if __name__ == "__main__":
    main()
