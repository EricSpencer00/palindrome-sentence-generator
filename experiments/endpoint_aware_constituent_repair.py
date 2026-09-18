"""Repair impossible palindrome endpoints before attempting interior edits.

The earlier local repair operator held an entire sentence opening fixed.  That
is fatal when its reversed spelling cannot occur at a grammatical ending (for
example, ``The`` requires an ending in ``eht``).  This experiment treats the
opening subject phrase and the terminal constituent as *jointly editable*.
It solves their letter debt from the outside in, carries an unmatched run
across word boundaries, and expands a failed patch to its next syntactic
ancestor on the following round.

The semantic plans are independently authored ordinary sentences.  They are
only construction material: a surface passing the solver is still checked by
the shared exclusion gate and then needs blinded human readers.  No catalogue
text, mirror phrase, repeated unit, or reader score is used to create or admit
an output.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
from math import prod
from pathlib import Path
import re
import sys
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.bidirectional_attested_span_mining import common_lexicon
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize


WORD = re.compile(r"[a-z]+")
ROUNDS = 8
BEAM = 64
PATCHES_PER_STATE = 32
MIN_LETTERS = 30
# The paper-facing long-output floor is 100 letters. The active repair method
# searches beyond it, rather than treating its former 120-letter ceiling as a
# definition of success.
MAX_LETTERS = 160


@dataclass(frozen=True)
class Plan:
    """One authored event and conservative synonymous constituent menus."""

    identifier: str
    event: str
    seed: str
    # Slots preserve a simple declarative grammar in left-to-right order.
    # Every option in a slot expresses the same role in the declared event.
    slots: tuple[tuple[str, ...], ...]


# These are written as complete ordinary sentences rather than retrieved from a
# palindrome collection.  The choice menus give the operator authority to
# replace a determiner, number, inflection, adjective, and terminal noun while
# keeping the stated event intact.  The lists are intentionally small: this is
# a reproducible repair run, not a claim that a finite grammar proves anything.
PLANS = (
    Plan("editors-drafts", "editors improve manuscript drafts",
         "Some careful editors revise old drafts.",
         (("some", "several"), ("careful", "patient"), ("editors", "writers"),
          ("revise", "edit"), ("old", "rough"), ("drafts", "texts"))),
    Plan("teachers-readers", "teachers help young readers",
         "Some patient teachers guide young readers.",
         (("some", "several"), ("patient", "calm"), ("teachers", "tutors"),
          ("guide", "teach"), ("young", "quiet"), ("readers", "pupils"))),
    Plan("nurses-wounds", "nurses examine minor wounds",
         "Those patient nurses examine small wounds.",
         (("those", "these"), ("patient", "calm"), ("nurses", "medics"),
          ("examine", "inspect"), ("small", "minor"), ("wounds", "injuries"))),
    Plan("bakers-bread", "bakers warm fresh bread",
         "Many quiet bakers warm fresh bread.",
         (("many", "some"), ("quiet", "calm"), ("bakers", "cooks"),
          ("warm", "heat"), ("fresh", "new"), ("bread", "loaves"))),
    Plan("artists-frames", "artists repair damaged frames",
         "Those local artists repair broken frames.",
         (("those", "these"), ("local", "nearby"), ("artists", "painters"),
          ("repair", "mend"), ("broken", "damaged"), ("frames", "pictures"))),
    Plan("sailors-maps", "sailors carry marked maps",
         "Our careful sailors carry marked maps.",
         (("our", "these"), ("careful", "steady"), ("sailors", "pilots"),
          ("carry", "bring"), ("marked", "folded"), ("maps", "charts"))),
    Plan("poets-verses", "poets write brief verses",
         "Several calm poets write vivid verses.",
         (("several", "some"), ("calm", "quiet"), ("poets", "writers"),
          ("write", "draft"), ("vivid", "short"), ("verses", "poems"))),
    Plan("farmers-tools", "farmers return lost tools",
         "Those young farmers return lost tools.",
         (("those", "these"), ("young", "local"), ("farmers", "workers"),
          ("return", "recover"), ("lost", "missing"), ("tools", "items"))),
    Plan("readers-notes", "readers find clear notes",
         "Some curious readers find clear notes.",
         (("some", "several"), ("curious", "careful"), ("readers", "students"),
          ("find", "notice"), ("clear", "brief"), ("notes", "letters"))),
    Plan("drivers-signs", "drivers notice road signs",
         "Those careful drivers notice road signs.",
         (("those", "these"), ("careful", "alert"), ("drivers", "riders"),
          ("notice", "watch"), ("road", "nearby"), ("signs", "markers"))),
    Plan("guards-gates", "guards close open gates",
         "Some quiet guards close open gates.",
         (("some", "several"), ("quiet", "calm"), ("guards", "keepers"),
          ("close", "secure"), ("open", "outer"), ("gates", "doors"))),
    Plan("students-questions", "students answer hard questions",
         "Many serious students answer hard questions.",
         (("many", "some"), ("serious", "eager"), ("students", "pupils"),
          ("answer", "solve"), ("hard", "tough"), ("questions", "problems"))),
    Plan("doctors-reports", "doctors review patient reports",
         "Those careful doctors review patient reports.",
         (("those", "these"), ("careful", "senior"), ("doctors", "medics"),
          ("review", "read"), ("patient", "recent"), ("reports", "records"))),
    Plan("workers-boxes", "workers move heavy boxes",
         "Some strong workers move heavy boxes.",
         (("some", "several"), ("strong", "steady"), ("workers", "movers"),
          ("move", "carry"), ("heavy", "large"), ("boxes", "crates"))),
    Plan("friends-letters", "friends send kind letters",
         "Those close friends send kind letters.",
         (("those", "these"), ("close", "kind"), ("friends", "neighbours"),
          ("send", "write"), ("kind", "brief"), ("letters", "notes"))),
    Plan("visitors-photos", "visitors take bright photographs",
         "Many curious visitors take bright photos.",
         (("many", "some"), ("curious", "eager"), ("visitors", "tourists"),
          ("take", "make"), ("bright", "clear"), ("photos", "pictures"))),
    Plan("we-warnings", "a group notices a small number of warnings",
         "We notice few warnings.",
         (("we",), ("notice", "observe"), ("few", "some"), ("warnings", "signals"))),
    Plan("we-maps", "a group reviews a small number of maps",
         "We review few maps.",
         (("we",), ("review", "read"), ("few", "some"), ("maps", "charts"))),
    Plan("no-baker-melon", "no baker preserves a melon",
         "No baker saves a melon.",
         (("no",), ("baker", "cook"), ("saves", "keeps"), ("a", "one"), ("melon", "lemon"))),
    Plan("no-artist-ribbon", "no artist retains a ribbon",
         "No artist keeps a ribbon.",
         (("no",), ("artist", "painter"), ("keeps", "saves"), ("a", "one"), ("ribbon", "token"))),
    Plan("no-tutor-lesson", "no tutor reads a lesson",
         "No tutor reads a lesson.",
         (("no",), ("tutor", "teacher"), ("reads", "reviews"), ("a", "one"), ("lesson", "passage"))),
    Plan("no-nurse-note", "no nurse writes a note",
         "No nurse writes a note.",
         (("no",), ("nurse", "medic"), ("writes", "drafts"), ("a", "one"), ("note", "memo"))),
    Plan("we-records", "a group checks few records",
         "We check few records.",
         (("we",), ("check", "review"), ("few", "some"), ("records", "reports"))),
    Plan("we-tools", "a group repairs few tools",
         "We repair few tools.",
         (("we",), ("repair", "mend"), ("few", "some"), ("tools", "items"))),
    Plan("we-doors", "a group closes few doors",
         "We close few doors.",
         (("we",), ("close", "secure"), ("few", "some"), ("doors", "gates"))),
    Plan("we-notes", "a group sends few notes",
         "We send few notes.",
         (("we",), ("send", "write"), ("few", "some"), ("notes", "letters"))),
    Plan("no-guard-gate", "no guard closes a gate",
         "No guard closes a gate.",
         (("no",), ("guard", "keeper"), ("closes", "secures"), ("a", "one"), ("gate", "door"))),
    Plan("no-driver-map", "no driver carries a map",
         "No driver carries a map.",
         (("no",), ("driver", "rider"), ("carries", "brings"), ("a", "one"), ("map", "chart"))),
    Plan("no-poet-verse", "no poet writes a verse",
         "No poet writes a verse.",
         (("no",), ("poet", "writer"), ("writes", "drafts"), ("a", "one"), ("verse", "poem"))),
    Plan("no-reader-letter", "no reader sends a letter",
         "No reader sends a letter.",
         (("no",), ("reader", "student"), ("sends", "writes"), ("a", "one"), ("letter", "note"))),
    Plan("we-pictures", "a group reviews few pictures",
         "We review few pictures.",
         (("we",), ("review", "inspect"), ("few", "some"), ("pictures", "photos"))),
    Plan("we-verses", "a group reads few verses",
         "We read few verses.",
         (("we",), ("read", "review"), ("few", "some"), ("verses", "poems"))),
)


def independent_two_pointer(text: str) -> dict[str, object]:
    """A deliberately independent letter-only exactness audit."""
    try:
        tape = normalize_letters(text)
    except (TypeError, ValueError):
        return {"normalized": "", "exact": False, "first_mismatch": None}
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            return {"normalized": tape, "exact": False,
                    "first_mismatch": [left, right]}
        left += 1
        right -= 1
    return {"normalized": tape, "exact": bool(tape), "first_mismatch": None}


def prefix_debt(left: str, right: str) -> dict[str, object]:
    """Intersect an opening and a terminal patch before interior expansion."""
    front = normalize_letters(left)
    terminal_reversed = normalize_letters(right)[::-1]
    width = min(len(front), len(terminal_reversed))
    mismatch = next((index for index in range(width)
                     if front[index] != terminal_reversed[index]), None)
    if mismatch is not None:
        return {"compatible": False, "matched_letters": mismatch,
                "residual_debt": None, "first_mismatch": mismatch}
    return {
        "compatible": True,
        "matched_letters": width,
        "residual_debt": (front[width:] if len(front) > width
                            else terminal_reversed[width:]),
        "debt_owner": "opening" if len(front) > width else "terminal",
        "first_mismatch": None,
    }


def _slot_options(slot: Sequence[str], vocabulary: set[str]) -> tuple[str, ...]:
    """Keep only observed, dictionary-backed surface forms in a patch menu."""
    return tuple(word for word in slot
                 if word in vocabulary and word.isascii() and word.isalpha())[:PATCHES_PER_STATE]


def _consume(debt: str, incoming: str) -> tuple[str, bool] | None:
    """Cancel a word spelling against current debt, carrying overflow."""
    width = min(len(debt), len(incoming))
    if debt[:width] != incoming[:width]:
        return None
    return ((debt[width:], False) if len(debt) > len(incoming)
            else (incoming[width:], True))


def solve_joint_slots(slots: Sequence[Sequence[str]], *, beam: int = BEAM,
                      state_cap: int = 20_000) -> tuple[list[tuple[str, ...]], dict[str, int]]:
    """Bounded exact-debt solve over complete grammatical constituent slots.

    The leftmost or rightmost unfilled slot is selected by the debt owner.  A
    right slot consumes its spelling in reverse, so word boundaries can differ
    across the palindrome centre.  This is a character constraint solver, not
    a word-order mirror generator.
    """
    menus = tuple(tuple(dict.fromkeys(slot))[:PATCHES_PER_STATE] for slot in slots)
    represented = prod(len(menu) for menu in menus)
    # lo, hi, debt, owner, selected slots.  owner 1 means the opening side
    # owns the unmatched letters; -1 means the terminal side owns them.
    frontier = [(0, len(menus) - 1, "", 1, ("",) * len(menus))]
    states = 0
    prefix_rejections = 0
    closures: list[tuple[str, ...]] = []
    while frontier and states < state_cap:
        next_frontier = []
        for lo, hi, debt, owner, selected in frontier:
            states += 1
            if lo > hi:
                if debt == debt[::-1]:
                    closures.append(selected)
                continue
            side = -owner if debt else 1
            index = lo if side == 1 else hi
            for word in menus[index]:
                incoming = word if side == 1 else word[::-1]
                result = _consume(debt, incoming)
                if result is None:
                    prefix_rejections += 1
                    continue
                remaining, overflow = result
                next_selected = list(selected)
                next_selected[index] = word
                next_frontier.append((lo + (side == 1), hi - (side == -1), remaining,
                                      side if overflow else owner, tuple(next_selected)))
        # A smaller debt is more repairable; this only orders expansion and
        # never admits text or assigns a readability score.
        next_frontier.sort(key=lambda state: (len(state[2]), state[4]))
        frontier = next_frontier[:beam]
    return sorted(set(closures)), {
        "represented_derivations": represented,
        "states_visited": states,
        "prefix_rejections": prefix_rejections,
        "state_cap": state_cap,
        "state_cap_hit": int(bool(frontier and states >= state_cap)),
    }


def _render(words: Iterable[str]) -> str:
    text = " ".join(words)
    return text[:1].upper() + text[1:] + "."


def _audit(text: str, catalogue: set[str]) -> dict[str, object]:
    two_pointer = independent_two_pointer(text)
    checks = mechanical_admission_checks(
        text, local_catalogue=catalogue, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS
    )
    return {
        "rendered": text,
        "normalized": two_pointer["normalized"],
        "length": len(str(two_pointer["normalized"])),
        "independent_two_pointer": two_pointer,
        "checks": checks,
        "rejection_codes": [name for name, passed in checks.items() if not passed],
        "mechanically_admitted": bool(two_pointer["exact"] and all(checks.values())),
    }


def _round_slots(plan: Plan, round_index: int, vocabulary: set[str]) -> tuple[tuple[str, ...], ...]:
    """Expand a failed patch to its next constituent ancestor each round."""
    # Round 0 holds the authored seed wording.  Subsequent rounds open one
    # additional slot at each end, then work inward; by round 7 all roles have
    # their conservative semantic alternatives.  This preserves the plan's
    # event while granting structural (not just local word) repair authority.
    slots = []
    width = min(len(plan.slots), 1 + round_index)
    for index, slot in enumerate(plan.slots):
        choices = slot if index < width or index >= len(plan.slots) - width else slot[:1]
        filtered = _slot_options(choices, vocabulary)
        if not filtered:
            raise ValueError(f"no observed surface forms for {plan.identifier}:{index}")
        slots.append(filtered)
    return tuple(slots)


def _endpoint_history(slots: Sequence[Sequence[str]]) -> list[dict[str, object]]:
    """Record all current opening/terminal constituent intersections."""
    front = slots[0]
    terminal = slots[-1]
    return [
        {"opening_patch": left, "terminal_patch": right,
         "intersection": prefix_debt(left, right)}
        for left, right in itertools.product(front, terminal)
    ]


def run(*, min_zipf: float = 3.0, rounds: int = ROUNDS, beam: int = BEAM,
        state_cap: int = 20_000) -> dict[str, object]:
    """Run all authored plans and retain every exact closure, including zero."""
    if rounds != ROUNDS:
        raise ValueError(f"this preregistered run has exactly {ROUNDS} repair rounds")
    vocabulary = common_lexicon(min_zipf)
    catalogue_path = ROOT / "data" / "known_palindromes.json"
    catalogue = set(json.loads(catalogue_path.read_text()))
    all_records: list[dict[str, object]] = []
    exact_records: list[dict[str, object]] = []
    for plan in PLANS:
        plan_rounds = []
        for round_index in range(rounds):
            slots = _round_slots(plan, round_index, vocabulary)
            closures, stats = solve_joint_slots(slots, beam=beam, state_cap=state_cap)
            # The initial ordinary surface is retained as a rendered repair
            # proposal, rather than suppressing failed material.  It has no
            # possibility of being presented to readers unless exactness and
            # the shared exclusion gate both pass.
            seed_text = _render(slot[0] for slot in slots)
            seed_audit = _audit(seed_text, catalogue)
            closure_audits = [_audit(_render(words), catalogue) for words in closures]
            exact_records.extend(closure_audits)
            plan_rounds.append({
                "round": round_index,
                "repair_operator": (
                    "expand_failed_opening_and_terminal_patch_to_next_syntactic_ancestor"
                    if round_index else "joint_opening_terminal_constituent_intersection"
                ),
                "slot_domains": [list(slot) for slot in slots],
                "endpoint_intersections": _endpoint_history(slots),
                "seed_proposal": seed_audit,
                "solver": stats,
                "exact_closures": closure_audits,
            })
        all_records.append({"id": plan.identifier, "event": plan.event, "authored_seed": plan.seed,
                            "repair_rounds": plan_rounds})
    admitted = [row for row in exact_records if row["mechanically_admitted"]]
    plans_payload = [
        {"id": plan.identifier, "event": plan.event, "seed": plan.seed, "slots": plan.slots}
        for plan in PLANS
    ]
    return {
        "status": "complete_endpoint_aware_joint_constituent_repair_run",
        "config": {"plans": len(PLANS), "rounds": rounds, "beam": beam,
                   "patches_per_state": PATCHES_PER_STATE, "state_cap": state_cap,
                   "min_zipf": min_zipf, "min_letters": MIN_LETTERS,
                   "max_letters": MAX_LETTERS},
        "provenance": {
            "plans_sha256": sha256(json.dumps(plans_payload, sort_keys=True).encode()).hexdigest(),
            "vocabulary_sha256": sha256("\n".join(sorted(vocabulary)).encode()).hexdigest(),
            "catalogue_sha256": sha256(catalogue_path.read_bytes()).hexdigest(),
            "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "surface_vocabulary": (
                "headwords plus Brown/wordfreq-observed forms filtered by the local "
                "regular-inflection dictionary gate"
            ),
        },
        "plans": all_records,
        "exact_closures": exact_records,
        "mechanically_admitted": admitted,
        "reader_facing_next_test": (
            "No reader package is created until a mechanically admitted output exists. For each "
            "such output, randomize the candidate, independently written intact prose expressing "
            "the same intended event, and a shuffled control; blind readers to condition and collect "
            "one-pass readability, grammatical completeness, and a free paraphrase."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-zipf", type=float, default=3.0)
    parser.add_argument("--beam", type=int, default=BEAM)
    parser.add_argument("--state-cap", type=int, default=20_000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(min_zipf=args.min_zipf, beam=args.beam, state_cap=args.state_cap)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "plans": len(result["plans"]),
                      "exact_closures": len(result["exact_closures"]),
                      "admitted": len(result["mechanically_admitted"])}, indent=2))


if __name__ == "__main__":
    main()
