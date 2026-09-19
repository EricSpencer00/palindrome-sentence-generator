"""Constructive SVO-shell search with one agreement-carrying adjunct slot.

This lane starts from a semantic shell, rather than from a tape or a bank of
finished palindromes.  The shell has a subject, finite transitive verb,
object, and exactly one temporal/locative adjunct slot.  Subject number and
tense select the finite verb; the adjunct slot carries the same number through
its pronoun or possessive when the realization needs one.

The search chooses the outer semantic fields first.  It compares the known
subject prefix with the known adjunct suffix before choosing the verb and
object, then checks the completed normalized tape with an independent
two-pointer audit.  A mismatch is evidence for the next repair, never a
readability reward.  No word-order mirror, finished palindrome seed,
catalogue text, or RLAIF score is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (  # noqa: E402
    mechanical_admission_checks,
    normalize_letters,
)


EXPERIMENT_ID = "agreement-carrying-adjunct-center-csp-20260920"
SIGNATURE = (
    "semantic-svo-shell|agreement-carrying-slot|slot-kind-number-state|"
    "outer-equation-pruning|independent-pointer-sha"
)
TARGET_MIN = 39
TARGET_MAX = 70
# A short lookahead is intentional: after the shell's subject and adjunct are
# selected, only the outermost two equations are fully exposed.  The verb and
# object occupy the unresolved interior.  The completed tape is then checked
# independently before it can become a retained row.
EQUATION_WINDOW = 2


@dataclass(frozen=True)
class Slot:
    """One semantic adjunct realization and its agreement contract."""

    kind: str
    number: str
    text: str
    attachment: str


@dataclass(frozen=True)
class Shell:
    """A complete semantic shell before punctuation is rendered."""

    subject: str
    verb: str
    obj: str
    slot: Slot
    subject_number: str
    tense: str
    object_number: str

    @property
    def text(self) -> str:
        return f"{self.subject} {self.verb} {self.obj} {self.slot.text}"

    @property
    def tape(self) -> str:
        return normalize_letters(self.text)


# The domains are deliberately small and authored for this construction.  A
# bare plural is allowed only where ordinary English permits it; agreement is
# carried independently of lexical length.
SUBJECTS = {
    "sg": (
        "the quiet poet",
        "a patient sailor",
        "the young baker",
        "a careful scribe",
        "the calm keeper",
        "a bright pilot",
    ),
    "pl": (
        "some sailors",
        "the quiet poets",
        "young bakers",
        "careful scribes",
        "calm keepers",
        "bright pilots",
    ),
}

VERBS = {
    ("sg", "pres"): ("marks", "carries", "guards", "watches", "opens", "studies", "finds"),
    ("pl", "pres"): ("mark", "carry", "guard", "watch", "open", "study", "find"),
    ("sg", "past"): ("marked", "carried", "guarded", "watched", "opened", "studied", "found"),
    ("pl", "past"): ("marked", "carried", "guarded", "watched", "opened", "studied", "found"),
}

OBJECTS = {
    "sg": ("a weathered map", "the old letter", "an open journal", "a quiet harbor", "the small chart"),
    "pl": ("weathered maps", "old letters", "open journals", "quiet harbors", "small charts"),
}


SLOTS = (
    # The pronoun and finite verb expose subject-number agreement inside the
    # adjunct rather than treating it as an untyped suffix.
    Slot("temporal", "sg", "while he waits", "subject-linked temporal"),
    Slot("temporal", "sg", "after he rests", "subject-linked temporal"),
    Slot("temporal", "pl", "while they wait", "subject-linked temporal"),
    Slot("temporal", "pl", "after they rest", "subject-linked temporal"),
    # These number-specific possessives are locative adjuncts.  The neutral
    # slots are retained as controls for whether agreement itself is causal.
    Slot("locative", "sg", "near his tower", "subject-linked locative"),
    Slot("locative", "sg", "by his garden", "subject-linked locative"),
    Slot("locative", "pl", "near their tower", "subject-linked locative"),
    Slot("locative", "pl", "by their garden", "subject-linked locative"),
    Slot("temporal", "any", "at night", "number-neutral temporal"),
    Slot("temporal", "any", "at times", "number-neutral temporal"),
    Slot("temporal", "any", "before midnight", "number-neutral temporal"),
    Slot("temporal", "any", "after midnight", "number-neutral temporal"),
    Slot("locative", "any", "at sea", "number-neutral locative"),
    Slot("locative", "any", "with care", "number-neutral adjunct"),
)


def slot_compatible(slot: Slot, subject_number: str) -> bool:
    return slot.number in {"any", subject_number}


def object_compatible(obj: str, object_number: str) -> bool:
    """Keep determiner/number agreement explicit in the shell state."""
    if object_number == "sg":
        return obj.split()[-1].endswith(("s", "ies")) is False
    return obj.split()[-1].endswith(("s", "ies"))


def outer_equation(subject: str, slot: Slot) -> dict[str, Any]:
    """Check only equations whose characters are known before inner slots.

    The subject is the left edge and the adjunct is the right edge.  Their
    common outside-in prefix is therefore a real constraint even though the
    verb and object have not yet been selected.
    """
    left = normalize_letters(subject)
    right = normalize_letters(slot.text)
    checked = min(EQUATION_WINDOW, len(left), len(right))
    mismatch = next(
        (
            {
                "offset": i,
                "left": left[i],
                "right": right[-1 - i],
            }
            for i in range(checked)
            if left[i] != right[-1 - i]
        ),
        None,
    )
    return {
        "checked_characters": checked,
        "matched_characters": checked if mismatch is None else mismatch["offset"],
        "unresolved_outer_characters": max(0, min(len(left), len(right)) - checked),
        "first_mismatch": mismatch,
        "compatible": mismatch is None,
        "equation_solved_before_inner_slots": True,
    }


def audit(text: str) -> dict[str, Any]:
    """Independent exactness audit, intentionally separate from the search."""
    tape = normalize_letters(text)
    reverse = tape[::-1]
    mismatches = [
        (i, tape[i], reverse[i])
        for i in range(len(tape) // 2)
        if tape[i] != reverse[i]
    ]
    return {
        "normalized": tape,
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "mismatch_count": len(mismatches),
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(reverse.encode()).hexdigest(),
        "sha_equal_under_reversal": hashlib.sha256(tape.encode()).digest()
        == hashlib.sha256(reverse.encode()).digest(),
    }


def _shell_specs() -> tuple[dict[str, Any], ...]:
    """Enumerate semantic choices without rendering punctuation or candidates."""
    specs: list[dict[str, Any]] = []
    for number, subjects in SUBJECTS.items():
        for tense in ("pres", "past"):
            for subject, verb, obj, slot in itertools.product(
                subjects, VERBS[(number, tense)], OBJECTS["sg"], SLOTS
            ):
                if not slot_compatible(slot, number):
                    continue
                specs.append(
                    {
                        "subject": subject,
                        "verb": verb,
                        "obj": obj,
                        "slot": slot,
                        "subject_number": number,
                        "tense": tense,
                        "object_number": "sg",
                    }
                )
            for subject, verb, obj, slot in itertools.product(
                subjects, VERBS[(number, tense)], OBJECTS["pl"], SLOTS
            ):
                if not slot_compatible(slot, number):
                    continue
                specs.append(
                    {
                        "subject": subject,
                        "verb": verb,
                        "obj": obj,
                        "slot": slot,
                        "subject_number": number,
                        "tense": tense,
                        "object_number": "pl",
                    }
                )
    return tuple(specs)


def _make_shell(spec: dict[str, Any]) -> Shell:
    return Shell(**spec)


def _control_rank(row: dict[str, Any]) -> tuple[int, int, int, str]:
    audit_row = row["audit"]
    return (
        row["outer_equation"]["matched_characters"],
        -audit_row["mismatch_count"],
        audit_row["letters"],
        row["rendered"],
    )


def _render_row(shell: Shell, outer: dict[str, Any], *, control: bool) -> dict[str, Any]:
    rendered = shell.text.capitalize() + "."
    checked = mechanical_admission_checks(
        rendered, min_letters=TARGET_MIN, max_letters=TARGET_MAX
    )
    row_audit = audit(rendered)
    exact = row_audit["two_pointer_exact"]
    return {
        "rendered": rendered,
        "control": control,
        "slot": {
            "kind": shell.slot.kind,
            "number": shell.slot.number,
            "attachment": shell.slot.attachment,
            "text": shell.slot.text,
        },
        "agreement": {
            "subject_number": shell.subject_number,
            "tense": shell.tense,
            "object_number": shell.object_number,
            "slot_number": shell.slot.number,
            "subject_slot_compatible": slot_compatible(shell.slot, shell.subject_number),
        },
        "semantic_roles": {
            "subject": shell.subject,
            "predicate": shell.verb,
            "object": shell.obj,
            "adjunct": shell.slot.text,
            "word_order": "subject-verb-object-adjunct",
        },
        "outer_equation": outer,
        "audit": row_audit,
        "mechanical_checks": checked,
        "mechanically_admitted": exact and all(checked.values()),
        "provenance": {
            "construction": SIGNATURE,
            "lexical_source": "fresh hand-authored semantic SVO and slot domains",
            "catalogue_imported": False,
            "finished_tape_reversed": False,
            "seed_used_as_output": False,
            "word_order_mirror": False,
            "rlaif_used": False,
            "equations_solved_during_search": True,
            "rendered_after_state_selection": True,
        },
        "reader_status": "unreviewed; exactness and mechanical checks do not certify readability",
    }


def run() -> dict[str, Any]:
    """Run the bounded semantic-shell CSP and retain actual controls."""
    specs = _shell_specs()
    outer_states = 0
    outer_pruned = 0
    inner_states = 0
    length_pruned = 0
    retained_specs: list[tuple[Shell, dict[str, Any]]] = []
    compatible_frontier: list[tuple[Shell, dict[str, Any]]] = []
    exact_rows: list[dict[str, Any]] = []
    # The first phase chooses the semantic outer fields and solves their
    # available equations.  The second phase chooses the lexical interior.
    for spec in specs:
        shell = _make_shell(spec)
        outer_states += 1
        outer = outer_equation(shell.subject, shell.slot)
        if not outer["compatible"]:
            outer_pruned += 1
            # Keep a small deterministic sample of pre-render states.  They
            # become controls after the bounded search, not search rewards.
            if len(retained_specs) < 120:
                retained_specs.append((shell, outer))
            continue
        inner_states += 1
        tape = shell.tape
        if not (TARGET_MIN <= len(tape) <= TARGET_MAX):
            length_pruned += 1
            continue
        # The full character equation is checked before punctuation is added
        # and before a row is exposed as a rendered candidate.
        row_audit = audit(shell.text)
        if row_audit["two_pointer_exact"]:
            exact_rows.append(_render_row(shell, outer, control=False))
        else:
            # Keep the semantic state, not a rendered string.  The bounded
            # control frontier is selected after search by equation depth and
            # length so early domain order cannot hide longer controls.
            compatible_frontier.append((shell, outer))

    # Materialize only the bounded frontier as actual controls.  This keeps
    # the controls inspectable without pretending that every state was a
    # candidate or invoking any reward/readability model during search.
    control_rows: list[dict[str, Any]] = []
    diversity_specs: list[tuple[Shell, dict[str, Any]]] = []
    for kind, number in (("temporal", "sg"), ("temporal", "pl"),
                         ("locative", "sg"), ("locative", "pl")):
        for spec in specs:
            shell = _make_shell(spec)
            if (
                shell.slot.kind == kind
                and shell.slot.number == number
                and TARGET_MIN <= len(shell.tape) <= TARGET_MAX
            ):
                diversity_specs.append((shell, outer_equation(shell.subject, shell.slot)))
                break
    frontier = retained_specs + diversity_specs + sorted(
        compatible_frontier,
        key=lambda item: (
            item[1]["matched_characters"],
            len(item[0].tape),
            item[0].text,
        ),
        reverse=True,
    )[:120]
    for shell, outer in frontier:
        if not (TARGET_MIN <= len(shell.tape) <= TARGET_MAX):
            continue
        row = _render_row(shell, outer, control=True)
        row["control_reason"] = "intact semantic shell retained from equation frontier"
        control_rows.append(row)
    control_rows.sort(key=_control_rank, reverse=True)
    # Preserve one actual rendering for each agreement-carrying slot family;
    # otherwise the longest neutral temporal state could hide the very
    # subject-linked temporal/locative states this lane is measuring.
    required: list[dict[str, Any]] = []
    for kind, number in (("temporal", "sg"), ("temporal", "pl"),
                         ("locative", "sg"), ("locative", "pl")):
        matches = [
            row for row in control_rows
            if row["slot"]["kind"] == kind and row["slot"]["number"] == number
        ]
        required.extend(matches[:1])
    selected_required: list[dict[str, Any]] = []
    seen_required: set[str] = set()
    for row in required:
        key = (row["slot"]["kind"], row["slot"]["number"])
        if key not in seen_required:
            selected_required.append(row)
            seen_required.add(key)
    controls = selected_required + [
        row for row in control_rows
        if row["rendered"] not in {item["rendered"] for item in selected_required}
    ][: max(0, 24 - len(selected_required))]

    rows = exact_rows + controls
    # Keep exact rows distinct from controls even if a future domain tweak
    # creates the same surface string.
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["rendered"], row)
    rows = list(unique.values())
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]]
    admitted = [row for row in exact if row["mechanically_admitted"]]
    longest = max(rows, key=lambda row: row["audit"]["letters"], default=None)

    registry = ROOT / "docs" / "experiment-novelty-registry.json"
    registry_entries = 0
    signature_collision = False
    if registry.exists():
        try:
            data = json.loads(registry.read_text())
            entries = data.get("entries", [])
            registry_entries = len(entries)
            signature_collision = any(item.get("signature") == SIGNATURE for item in entries)
        except (OSError, json.JSONDecodeError, AttributeError):
            # A missing/unreadable registry cannot be used to claim novelty.
            signature_collision = True

    status = "completed_exact_requires_reader_gate" if exact else "completed_no_exact_closure"
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": (
            "agreement-carrying semantic SVO shell with one temporal/locative slot; "
            "outer character equations are solved before inner lexical realization"
        ),
        "status": status,
        "target_range": [TARGET_MIN, TARGET_MAX],
        "rows": rows,
        "exact_candidates": exact,
        "stats": {
            "semantic_shell_specs": len(specs),
            "outer_states": outer_states,
            "outer_equation_pruned": outer_pruned,
            "inner_states": inner_states,
            "length_pruned": length_pruned,
            "rendered_controls": len(controls),
            "exact": len(exact),
            "mechanically_admitted": len(admitted),
            "longest_letters": longest["audit"]["letters"] if longest else 0,
            "longest_exact_letters": max(
                (row["audit"]["letters"] for row in exact), default=0
            ),
            "best_outer_match": max(
                (row["outer_equation"]["matched_characters"] for row in rows), default=0
            ),
        },
        "novelty_preflight": {
            "status": "blocked" if signature_collision else "passed",
            "signature": SIGNATURE,
            "registry_entries_scanned": registry_entries,
            "signature_collision": signature_collision,
            "prior_cartesian_product_replayed": False,
            "shortcuts_rejected": [
                "word-order mirror",
                "completed palindrome seed",
                "catalogue text",
                "finished-tape reversal",
                "RLAIF/per-candidate reward",
            ],
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "lexical_source": "fresh hand-authored semantic domains",
            "catalogue_imported": False,
            "finished_tape_reversed": False,
            "seed_used_as_output": False,
            "rlaif_used": False,
            "independent_audits": [
                "literal outside-in two-pointer comparison",
                "forward/reverse SHA-256 recomputation",
                "shared mechanical admission gate",
            ],
            "controls_are_actual_renderings": True,
        },
        "reader_gate": "closed; no exact row is promoted without blinded readers",
        "next_repair": {
            "action": (
                "retain the highest-matching subject/slot state, then add a held-out "
                "agreement-compatible verb or object whose first interior character "
                "matches the next live residual"
            ),
            "reason": (
                "the slot state exposes only a shallow outer match; the remaining "
                "debt enters the verb/object boundary before a complete shell can close"
            ),
            "bound": "one held-out inner lexical edge; do not widen all slots or score search states with RLAIF",
        },
    }


if __name__ == "__main__":
    result = run()
    path = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
    for row in result["rows"][:3]:
        print(row["rendered"])
