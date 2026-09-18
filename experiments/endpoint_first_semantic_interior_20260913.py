"""Endpoint-first exact search with an editable semantic sentence interior.

The constructor chooses complete grammatical opening and terminal constituents
before it expands the event-bearing interior.  An endpoint pair is discarded
as soon as its known outside letters disagree; compatible pairs then grow by
choosing role-labelled interior constituents from both ends.  Thus a word may
cross a shifted boundary at any point, while exactness is a search invariant,
not a post-hoc splice.

This is a bounded experiment.  It does not use the palindrome catalogue,
relexicalize a known example, mirror words, or repeat a sentence unit.  Every
complete rendered path is audited independently and retained, including hard
rejections.  Passing the shared mechanical gate is not a readability claim;
reader evidence is required before promotion.
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

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

MIN_LETTERS = 100
MAX_LETTERS = 180
DEFAULT_STATE_CAP = 30_000
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class InteriorSlot:
    role: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class SentencePlan:
    identifier: str
    event: str
    source: str
    openings: tuple[str, ...]
    interiors: tuple[InteriorSlot, ...]
    terminals: tuple[str, ...]


# Each menu is a complete role-bearing surface phrase.  The choices in an
# interior slot preserve one event frame; they are not a free word dictionary.
# Sources and alternatives were authored for this experiment.
PLANS = (
    SentencePlan(
        "editors-archives",
        "careful editors revise old manuscripts before archivists catalogue them",
        "We revise weathered manuscripts while patient archivists catalogue restored volumes for the history display after review.",
        ("careful editors", "patient editors", "experienced editors", "we"),
        (
            InteriorSlot("matrix_finite", ("revise", "edit", "repair")),
            InteriorSlot("matrix_object", ("weathered manuscripts", "old working manuscripts", "damaged paper manuscripts")),
            InteriorSlot("event_connector", ("while", "as", "because")),
            InteriorSlot("reason_subject", ("patient archivists", "quiet archivists", "museum archivists")),
            InteriorSlot("reason_finite", ("catalogue", "label", "preserve")),
            InteriorSlot("reason_object", ("restored volumes", "repaired archive volumes", "old library volumes")),
            InteriorSlot("reason_context", ("for the history display", "before the public opening", "near the museum archive")),
        ),
        ("after review", "after the public lecture", "during morning work"),
    ),
    SentencePlan(
        "nurses-ward",
        "trained nurses carry clean supplies while doctors prepare the ward",
        "We carry sterile dressings while senior doctors prepare quiet rooms for recovering patients after review.",
        ("trained nurses", "skilled nurses", "attentive nurses", "we"),
        (
            InteriorSlot("matrix_finite", ("carry", "bring", "transport")),
            InteriorSlot("matrix_object", ("sterile dressings", "clean medical dressings", "fresh treatment supplies")),
            InteriorSlot("event_connector", ("while", "as", "because")),
            InteriorSlot("reason_subject", ("senior doctors", "careful doctors", "ward doctors")),
            InteriorSlot("reason_finite", ("prepare", "arrange", "inspect")),
            InteriorSlot("reason_object", ("quiet rooms", "clean treatment rooms", "new patient rooms")),
            InteriorSlot("reason_context", ("for recovering patients", "before the evening rounds", "during the afternoon shift")),
        ),
        ("after review", "before the evening rounds", "near the quiet ward"),
    ),
    SentencePlan(
        "gardeners-shelter",
        "careful gardeners restore a courtyard while volunteers prepare shelter",
        "We restore a sheltered courtyard while helpful volunteers prepare clean beds for new visitors after review.",
        ("careful gardeners", "steady gardeners", "local gardeners", "we"),
        (
            InteriorSlot("matrix_finite", ("restore", "repair", "prepare")),
            InteriorSlot("matrix_object", ("a sheltered courtyard", "the quiet garden courtyard", "a damaged stone courtyard")),
            InteriorSlot("event_connector", ("while", "as", "because")),
            InteriorSlot("reason_subject", ("helpful volunteers", "patient volunteers", "shelter volunteers")),
            InteriorSlot("reason_finite", ("prepare", "arrange", "clean")),
            InteriorSlot("reason_object", ("clean beds", "fresh visitor beds", "quiet sleeping rooms")),
            InteriorSlot("reason_context", ("for new visitors", "before the evening gathering", "near the community hall")),
        ),
        ("after review", "near the community hall", "after the garden meeting"),
    ),
)


def independent_two_pointer(text: str) -> dict[str, object]:
    """Recompute the ASCII tape with a standalone two-pointer audit."""
    try:
        tape = normalize_letters(text)
    except (TypeError, ValueError):
        return {"normalized": "", "letters": 0, "exact": False, "first_mismatch": None}
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            return {"normalized": tape, "letters": len(tape), "exact": False,
                    "first_mismatch": [left, right]}
        left += 1
        right -= 1
    return {"normalized": tape, "letters": len(tape), "exact": bool(tape),
            "first_mismatch": None}


def prefix_debt(opening: str, terminal: str) -> dict[str, object]:
    """Check known endpoint letters, retaining the unequal residual run."""
    left = normalize_letters(opening)
    right = normalize_letters(terminal)[::-1]
    width = min(len(left), len(right))
    mismatch = next((i for i in range(width) if left[i] != right[i]), None)
    if mismatch is not None:
        return {"compatible": False, "matched_letters": mismatch,
                "residual_debt": None, "first_mismatch": mismatch}
    residual = left[width:] if len(left) > width else right[width:]
    return {"compatible": True, "matched_letters": width,
            "residual_debt": residual,
            "debt_owner": "opening" if len(left) > width else "terminal",
            "first_mismatch": None}


def endpoint_pairs(plan: SentencePlan) -> list[dict[str, object]]:
    """Enumerate endpoint-compatible complete surface forms first."""
    rows = []
    for opening, terminal in itertools.product(plan.openings, plan.terminals):
        debt = prefix_debt(opening, terminal)
        rows.append({"opening": opening, "terminal": terminal, "intersection": debt,
                     "compatible": bool(debt["compatible"])})
    return rows


def _boundary_compatible(left_boundary: str, right_boundary: str) -> bool:
    left = normalize_letters(left_boundary)
    right = normalize_letters(right_boundary)[::-1]
    return left[:min(len(left), len(right))] == right[:min(len(left), len(right))]


def _render(opening: str, interior: Iterable[str], terminal: str) -> str:
    words = " ".join((opening, *interior, terminal)).strip()
    return words[:1].upper() + words[1:] + "."


def parse_surface_witness(plan: SentencePlan, text: str) -> dict[str, object]:
    """Replay the complete plan independently from the constructor state."""
    normalized = text.strip()
    matches = []
    for opening, choices, terminal in itertools.product(
        plan.openings,
        itertools.product(*(slot.options for slot in plan.interiors)),
        plan.terminals,
    ):
        if _render(opening, choices, terminal) == normalized:
            matches.append({"opening": opening, "interior": list(choices), "terminal": terminal,
                            "roles": [slot.role for slot in plan.interiors]})
    return {"independent_surface_parse": bool(matches), "derivations": matches,
            "complete_sentence": bool(matches), "plan_id": plan.identifier}


def search_plan(plan: SentencePlan, *, state_cap: int = DEFAULT_STATE_CAP) -> dict[str, object]:
    """Search each compatible endpoint pair while checking closure on every expansion."""
    if state_cap < 1:
        raise ValueError("state_cap must be positive")
    all_candidates: list[dict[str, object]] = []
    endpoint_history = endpoint_pairs(plan)
    states = 0
    prefix_rejections = 0
    endpoint_rejections = sum(not row["compatible"] for row in endpoint_history)
    rejections: list[dict[str, object]] = []
    frontier = []
    state_cap_hit = False
    for endpoint in endpoint_history:
        if not endpoint["compatible"]:
            continue
        opening, terminal = str(endpoint["opening"]), str(endpoint["terminal"])
        frontier = [(0, len(plan.interiors) - 1, (), (), opening, terminal)]
        while frontier and states < state_cap:
            next_frontier = []
            for lo, hi, left_words, right_words, left_boundary, right_boundary in frontier:
                states += 1
                if lo > hi:
                    interior = left_words + tuple(reversed(right_words))
                    rendered = _render(opening, interior, terminal)
                    all_candidates.append({"rendered": rendered, "kind": "exact_closure", "opening": opening,
                        "terminal": terminal, "interior": list(interior),
                        "endpoint_intersection": endpoint["intersection"],
                        "search_state": {"states": states, "complete": True}})
                    continue
                if lo == hi:
                    options = plan.interiors[lo].options
                    for option in options:
                        candidate_left = left_boundary + " " + option
                        candidate_right = right_boundary
                        if _boundary_compatible(candidate_left, candidate_right):
                            next_frontier.append((lo + 1, hi - 1, left_words + (option,), right_words,
                                                  candidate_left, candidate_right))
                        else:
                            prefix_rejections += 1
                            interior = left_words + (option,) + right_words
                            interior += tuple(slot.options[0] for slot in plan.interiors[lo + 1:hi + 1])
                            rejections.append({"rendered": _render(opening, interior, terminal),
                                               "kind": "partial_boundary_rejection",
                                               "rejection_code": "partial_boundary_mismatch",
                                               "opening": opening, "terminal": terminal,
                                               "interior": list(interior),
                                               "endpoint_intersection": endpoint["intersection"]})
                    continue
                left_options = plan.interiors[lo].options
                right_options = plan.interiors[hi].options
                for left_option, right_option in itertools.product(left_options, right_options):
                    new_left = left_boundary + " " + left_option
                    new_right = right_option + " " + right_boundary
                    if not _boundary_compatible(new_left, new_right):
                        prefix_rejections += 1
                        interior = left_words + (left_option,)
                        interior += tuple(slot.options[0] for slot in plan.interiors[lo + 1:hi])
                        interior += (right_option,) + right_words
                        rejections.append({"rendered": _render(opening, interior, terminal),
                                           "kind": "partial_boundary_rejection",
                                           "rejection_code": "partial_boundary_mismatch",
                                           "opening": opening, "terminal": terminal,
                                           "interior": list(interior),
                                           "endpoint_intersection": endpoint["intersection"]})
                        continue
                    next_frontier.append((lo + 1, hi - 1, left_words + (left_option,),
                                          right_words + (right_option,), new_left, new_right))
            frontier = next_frontier
        state_cap_hit = state_cap_hit or bool(frontier and states >= state_cap)
    unique = {}
    for row in all_candidates + rejections:
        unique.setdefault(row["rendered"], row)
    return {"endpoint_history": endpoint_history, "candidates": list(unique.values()),
            "rejections": rejections,
            "stats": {"states_visited": states, "endpoint_pairs": len(endpoint_history),
                       "endpoint_rejections": endpoint_rejections,
                       "partial_boundary_rejections": prefix_rejections,
                       "state_cap": state_cap, "state_cap_hit": state_cap_hit},
            "exhausted": not state_cap_hit}


def audit_candidate(plan: SentencePlan, rendered: str) -> dict[str, object]:
    independent = independent_two_pointer(rendered)
    checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    witness = parse_surface_witness(plan, rendered)
    checks["independent_exact_audit"] = bool(independent["exact"])
    checks["complete_grammar_witness"] = bool(witness["independent_surface_parse"])
    return {"rendered": rendered, "independent_two_pointer": independent,
            "sentence_witness": witness, "current_central_admission": checks,
            "rejection_codes": [key for key, value in checks.items() if not value],
            "mechanically_admitted": bool(all(checks.values())),
            "reader_status": "No independent reader evidence; no readability claim."}


def run(*, state_cap: int = DEFAULT_STATE_CAP) -> dict[str, object]:
    """Run the frozen plans and retain every complete rendered candidate."""
    plans = []
    records = []
    for plan in PLANS:
        search = search_plan(plan, state_cap=state_cap)
        candidate_records = []
        for candidate in search["candidates"]:
            audited = audit_candidate(plan, str(candidate["rendered"]))
            candidate_records.append({**candidate, "audit": audited})
            records.append({"source_id": plan.identifier, **candidate, "audit": audited})
        source_witness = parse_surface_witness(plan, plan.source)
        source_audit = audit_candidate(plan, plan.source)
        plans.append({"source_id": plan.identifier, "event": plan.event,
                      "source_sentence": plan.source, "source_control": source_audit,
                      "endpoint_history": search["endpoint_history"],
                      "search": search["stats"], "exhausted": search["exhausted"],
                      "candidates": candidate_records, "source_witness": source_witness})
    admitted = [row for row in records if row["audit"]["mechanically_admitted"]]
    payload = [{"id": p.identifier, "event": p.event, "source": p.source,
                "openings": p.openings, "interiors": [s.__dict__ for s in p.interiors],
                "terminals": p.terminals} for p in PLANS]
    return {"status": "complete_endpoint_first_semantic_interior_run",
            "config": {"plans": len(PLANS), "state_cap": state_cap,
                       "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
                       "endpoint_forms_selected_before_interior": True,
                       "exact_closure_checked_during_search": True,
                       "catalogue_used_for_generation": False},
            "provenance": {"plans_sha256": sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                           "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "construction_material": "Authored endpoint constituents and role-labelled semantic interiors; no catalogue, mirror pair, word reflection, or repeated unit.",
                           "central_admission": "llm_palindrome.admission.mechanical_admission_checks (unchanged)"},
            "plans": plans, "records": records, "mechanically_admitted": admitted,
            "readable_survivors": [],
            "reader_facing_next_test": (
                "No reader package is warranted while this frozen inventory has zero mechanically admitted outputs. "
                "The concrete next repair is to author additional endpoint-compatible subject and terminal adjunct "
                "forms with varied clause valency, then rerun this same exact bounded search. If any output clears "
                "the shared gate, blind it against independently authored intact prose and shuffled controls and "
                "collect one-pass readability, grammatical completeness, and paraphrase ratings." )}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--state-cap", type=int, default=DEFAULT_STATE_CAP)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(state_cap=args.state_cap)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(result["records"]),
                      "admitted": len(result["mechanically_admitted"])}, indent=2))


if __name__ == "__main__":
    main()
