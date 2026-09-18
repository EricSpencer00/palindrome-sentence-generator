"""Constituent-level paired insertion/deletion alignment.

The coupled projection experiment only changed characters while retaining a
fixed-length tape.  This operator keeps complete independently authored
subject, predicate, argument, and adjunct constituents, then explores a small
lattice in which a constituent can be replaced by another authored constituent
or inserted/deleted on either side.  The two clauses are never obtained by
reversing one another.  A pair is a proposal only after both clauses have been
rendered; an independent two-pointer audit and the unchanged central admission
gate decide whether it survives.

This is a bounded construction experiment, not a readability scorer.  Every
rendered proposal (including ordinary non-palindromic controls and hard-gate
rejections) is retained in the output.  A zero survivor rules out this frozen
constituent inventory and points to a concrete next repair; it does not rule
out constituent alignment generally.
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
DEFAULT_PROPOSALS_PER_PLAN = 160
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class Constituent:
    """A complete grammatical constituent with a small authored menu."""

    role: str
    options: tuple[str, ...]
    optional: bool = False


@dataclass(frozen=True)
class ClausePlan:
    identifier: str
    event: str
    left_source: str
    right_source: str
    left: tuple[Constituent, ...]
    right: tuple[Constituent, ...]


# The left and right clauses were authored independently.  The menus are
# complete phrases, not character substitutions.  Optional adjuncts make the
# paired insertion/deletion lattice explicit while preserving an intact core
# clause when an adjunct is absent.
PLANS = (
    ClausePlan(
        "editors-archives",
        "editors revise manuscripts while archivists label restored volumes",
        "Careful editors revise weathered manuscripts before morning meetings.",
        "Quiet archivists label restored volumes after afternoon lectures.",
        (
            Constituent("subject", ("careful editors", "patient editors")),
            Constituent("predicate", ("revise", "edit")),
            Constituent("argument", ("weathered manuscripts", "old manuscripts")),
            Constituent("adjunct", ("before morning meetings", "before noon meetings"), True),
        ),
        (
            Constituent("subject", ("quiet archivists", "calm archivists")),
            Constituent("predicate", ("label", "mark")),
            Constituent("argument", ("restored volumes", "repaired volumes")),
            Constituent("adjunct", ("after afternoon lectures", "after public lectures"), True),
        ),
    ),
    ClausePlan(
        "gardeners-shelter",
        "gardeners prepare a shelter while volunteers arrange supplies",
        "Steady gardeners prepare a sheltered courtyard beside the clinic.",
        "Helpful volunteers arrange clean supplies near the entrance.",
        (
            Constituent("subject", ("steady gardeners", "careful gardeners")),
            Constituent("predicate", ("prepare", "restore")),
            Constituent("argument", ("a sheltered courtyard", "the quiet courtyard")),
            Constituent("adjunct", ("beside the clinic", "behind the clinic"), True),
        ),
        (
            Constituent("subject", ("helpful volunteers", "patient volunteers")),
            Constituent("predicate", ("arrange", "sort")),
            Constituent("argument", ("clean supplies", "fresh supplies")),
            Constituent("adjunct", ("near the entrance", "near the office"), True),
        ),
    ),
    ClausePlan(
        "teachers-readers",
        "teachers guide young readers while librarians prepare lessons",
        "Patient teachers guide curious readers through difficult history lessons.",
        "Friendly librarians prepare useful examples for evening classes.",
        (
            Constituent("subject", ("patient teachers", "calm teachers")),
            Constituent("predicate", ("guide", "help")),
            Constituent("argument", ("curious readers", "young readers")),
            Constituent("adjunct", ("through difficult history lessons", "through new history lessons"), True),
        ),
        (
            Constituent("subject", ("friendly librarians", "helpful librarians")),
            Constituent("predicate", ("prepare", "collect")),
            Constituent("argument", ("useful examples", "clear examples")),
            Constituent("adjunct", ("for evening classes", "for weekly classes"), True),
        ),
    ),
    ClausePlan(
        "mechanics-bicycles",
        "mechanics repair bicycles while cyclists choose safer routes",
        "Experienced mechanics repair damaged bicycles beside the busy workshop.",
        "Local cyclists choose safer routes around the crowded market.",
        (
            Constituent("subject", ("experienced mechanics", "careful mechanics")),
            Constituent("predicate", ("repair", "mend")),
            Constituent("argument", ("damaged bicycles", "broken bicycles")),
            Constituent("adjunct", ("beside the busy workshop", "near the busy workshop"), True),
        ),
        (
            Constituent("subject", ("local cyclists", "daily cyclists")),
            Constituent("predicate", ("choose", "find")),
            Constituent("argument", ("safer routes", "quieter routes")),
            Constituent("adjunct", ("around the crowded market", "around the open market"), True),
        ),
    ),
    ClausePlan(
        "nurses-patients",
        "nurses examine patients while doctors explain treatment plans",
        "Attentive nurses examine recovering patients during the quiet afternoon.",
        "Senior doctors explain practical treatment plans beside the ward.",
        (
            Constituent("subject", ("attentive nurses", "patient nurses")),
            Constituent("predicate", ("examine", "observe")),
            Constituent("argument", ("recovering patients", "resting patients")),
            Constituent("adjunct", ("during the quiet afternoon", "during the calm afternoon"), True),
        ),
        (
            Constituent("subject", ("senior doctors", "careful doctors")),
            Constituent("predicate", ("explain", "describe")),
            Constituent("argument", ("practical treatment plans", "simple treatment plans")),
            Constituent("adjunct", ("beside the ward", "inside the ward"), True),
        ),
    ),
    ClausePlan(
        "writers-readers",
        "writers share essays while readers discuss difficult questions",
        "Thoughtful writers share finished essays with patient readers at noon.",
        "Curious readers discuss difficult questions after the public seminar.",
        (
            Constituent("subject", ("thoughtful writers", "careful writers")),
            Constituent("predicate", ("share", "send")),
            Constituent("argument", ("finished essays", "recent essays")),
            Constituent("adjunct", ("with patient readers", "with eager readers"), True),
        ),
        (
            Constituent("subject", ("curious readers", "thoughtful readers")),
            Constituent("predicate", ("discuss", "consider")),
            Constituent("argument", ("difficult questions", "important questions")),
            Constituent("adjunct", ("after the public seminar", "after the evening seminar"), True),
        ),
    ),
)


def independent_two_pointer(text: str) -> dict[str, object]:
    """Recompute ASCII letter symmetry without constructor helpers."""
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


def _options(constituent: Constituent, include_optional: bool) -> tuple[str | None, ...]:
    if constituent.optional:
        return ((None,) + constituent.options) if include_optional else (None,)
    return constituent.options


def paired_constituent_variants(
    plan: ClausePlan, *, limit: int = DEFAULT_PROPOSALS_PER_PLAN
) -> list[dict[str, object]]:
    """Enumerate bounded whole-constituent pair edits, including gaps.

    A ``None`` on one side is a deletion there; a non-``None`` constituent
    against ``None`` on the other side is an insertion.  All non-gap choices
    are complete authored phrases.  This lattice is the construction method;
    exact palindrome status is intentionally deferred to the independent
    rendered audit.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    rows: list[dict[str, object]] = []
    for left_choices, right_choices in itertools.product(
        itertools.product(*(_options(slot, True) for slot in plan.left)),
        itertools.product(*(_options(slot, True) for slot in plan.right)),
    ):
        # Preserve a complete clause core. Only optional adjuncts may be gaps.
        if any(value is None for value, slot in zip(left_choices, plan.left) if not slot.optional):
            continue
        if any(value is None for value, slot in zip(right_choices, plan.right) if not slot.optional):
            continue
        left_parts = tuple(value for value in left_choices if value)
        right_parts = tuple(value for value in right_choices if value)
        if not left_parts or not right_parts:
            continue
        operations = []
        for side, choices, slots in (("left", left_choices, plan.left), ("right", right_choices, plan.right)):
            for slot, value in zip(slots, choices):
                if value is None and slot.optional:
                    operations.append({"side": side, "operation": "delete_constituent",
                                       "role": slot.role, "text": slot.options[0]})
                elif value is not None and slot.optional and value != slot.options[0]:
                    operations.append({"side": side, "operation": "insert_constituent",
                                       "role": slot.role, "text": value})
                elif value is not None and value != slot.options[0]:
                    operations.append({"side": side, "operation": "replace_constituent",
                                       "role": slot.role, "text": value})
        if not operations:
            operator = "identity_constituents"
        elif any(op["operation"] == "insert_constituent" for op in operations) and any(
            op["operation"] == "delete_constituent" for op in operations
        ):
            operator = "paired_insert_delete_constituents"
        else:
            operator = "whole_constituent_menu_edit"
        rendered = _render(left_parts, right_parts)
        rows.append({
            "text": rendered,
            "left_constituents": list(left_parts),
            "right_constituents": list(right_parts),
            "operations": operations,
            "operator": operator,
            "preserved_constituent_count": sum(
                value == slot.options[0]
                for choices, slots in ((left_choices, plan.left), (right_choices, plan.right))
                for value, slot in zip(choices, slots) if value is not None
            ),
        })
        if len(rows) >= limit:
            break
    return rows


def _render(left: Iterable[str], right: Iterable[str]) -> str:
    """Render two independent clauses as one naturally punctuated proposal."""
    opening = " ".join(left).capitalize()
    terminal = " ".join(right).lower()
    return f"{opening}; {terminal}."


def audit_proposal(text: str) -> dict[str, object]:
    pointer = independent_two_pointer(text)
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    return {
        "rendered": text,
        "length": pointer["letters"],
        "independent_two_pointer": pointer,
        "current_central_admission": checks,
        "rejection_codes": [key for key, passed in checks.items() if not passed],
        "mechanically_admitted": bool(pointer["exact"] and all(checks.values())),
        "reader_status": "No human reader test was run; this record makes no readability claim.",
    }


def run(*, proposals_per_plan: int = DEFAULT_PROPOSALS_PER_PLAN) -> dict[str, object]:
    """Run the fixed authored inventory and retain every bounded proposal."""
    if proposals_per_plan < 1:
        raise ValueError("proposals_per_plan must be positive")
    records: list[dict[str, object]] = []
    for plan in PLANS:
        proposals = paired_constituent_variants(plan, limit=proposals_per_plan)
        for index, proposal in enumerate(proposals):
            records.append({
                "source_id": plan.identifier,
                "source_event": plan.event,
                "source_clauses": {
                    "left": plan.left_source,
                    "right": plan.right_source,
                    "authorship": "independently authored clause pair; right was not derived by reversing left",
                },
                "proposal_index": index,
                **proposal,
                "audit": audit_proposal(str(proposal["text"])),
            })
    admitted = [row for row in records if row["audit"]["mechanically_admitted"]]
    source_payload = [
        {"id": plan.identifier, "event": plan.event, "left_source": plan.left_source,
         "right_source": plan.right_source,
         "left": [slot.__dict__ for slot in plan.left],
         "right": [slot.__dict__ for slot in plan.right]}
        for plan in PLANS
    ]
    return {
        "status": "complete_bounded_constituent_alignment_repair_run",
        "config": {"plans": len(PLANS), "proposals_per_plan": proposals_per_plan,
                   "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
                   "operator": "paired whole-constituent insertion/deletion plus authored menu choices"},
        "provenance": {
            "source_clauses_sha256": sha256(json.dumps(source_payload, sort_keys=True).encode()).hexdigest(),
            "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "construction_material": "Six task-local independently authored clause pairs; no palindrome catalogue text, mirror pairs, or relexicalization inventory was used.",
            "central_admission": "llm_palindrome.admission.mechanical_admission_checks with its unchanged repository defaults",
        },
        "records": records,
        "mechanically_admitted": admitted,
        "readable_survivors": [],
        "reader_facing_next_test": (
            "No reader package is warranted while this inventory has zero mechanically admitted outputs. "
            "The concrete next repair is to author a second menu for each predicate and argument that "
            "permits constituent splitting/merging (for example, a prepositional argument versus an "
            "adjective+noun argument), then rerun the same bounded paired insertion/deletion audit. "
            "If a candidate clears the unchanged gate, blind it against independently written clause "
            "controls and shuffled controls before collecting readability, grammar, and paraphrase ratings."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--proposals-per-plan", type=int, default=DEFAULT_PROPOSALS_PER_PLAN)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(proposals_per_plan=args.proposals_per_plan)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(result["records"]),
                      "admitted": len(result["mechanically_admitted"])}, indent=2))


if __name__ == "__main__":
    main()
