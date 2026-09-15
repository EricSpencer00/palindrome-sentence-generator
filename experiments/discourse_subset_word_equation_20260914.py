"""Solve whole-discourse palindrome equations over original complete clauses.

Earlier clause-pair intersection asks whether reverse(A) equals B. Here an
unmatched suffix can cross arbitrarily many complete sentence boundaries:
A B C ... = reverse(... X Y Z). Sentences remain intact and each may occur
once. The search chooses their order and subset, without requiring any source
sentence to have a reversible partner. Every bank describes a single scene
using independent observations, making order variation grammatically licensed.

This bounded construction experiment has no language-model scorer. Exactness
does not confer readability, and no independent reader outcome is invented.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks


def letters(text: str) -> str:
    return "".join(char.lower() for char in text if char.isascii() and char.isalpha())


def independent_audit(text: str) -> dict:
    tape = []
    unsupported = []
    for char in text:
        code = ord(char)
        if 65 <= code <= 90:
            tape.append(chr(code + 32))
        elif 97 <= code <= 122:
            tape.append(char)
        elif char.isalpha():
            unsupported.append(char)
    bad = []
    lo, hi = 0, len(tape) - 1
    while lo < hi:
        if tape[lo] != tape[hi]:
            bad.append([lo, hi])
        lo += 1
        hi -= 1
    return {"normalized": "".join(tape), "letters": len(tape),
            "mismatched_pairs": bad, "mismatched_pair_count": len(bad),
            "unsupported_alphabetic_characters": unsupported,
            "exact": bool(tape) and not bad and not unsupported}


@dataclass(frozen=True)
class Scene:
    identifier: str
    description: str
    clauses: tuple[str, ...]


# Authored observation sentences, not palindrome material or retrieved prose.
# An individual sentence may be omitted; no pronoun depends on another item.
# Different sentences concern the same scene, but coherence is not scored.
SCENES = (
    Scene("studio", "Observations of people preparing a shared art studio.", (
        "An artist sketches.", "Some painters chat.", "Several students draw.",
        "Diana mixes colors.", "Lisa trims paper.", "Nadia labels portraits.",
        "An assistant sorts brushes.", "Some visitors inspect the murals.",
        "Clean sheets cover the tables.", "Light enters through tall windows.",
        "Music fills the room.", "Fresh flowers stand beside the easel.",
        "The instructor praises Diana.", "A volunteer helps Lisa.",
        "Quiet observers thank Nadia.", "Children admire the painted door.",
        "Staff arrange the chairs.", "One apprentice folds cloth.",
        "A conservator studies the canvas.", "The last guests depart.",
    )),
    Scene("workshop", "Independent observations inside a working repair shop.", (
        "Some mechanics work.", "An apprentice listens.", "Several engines idle.",
        "Clean tools hang beside the bench.", "Fresh parts fill the boxes.",
        "A driver tests the brakes.", "Diana checks tire pressure.",
        "Lisa repairs the pump.", "Nadia tightens bolts.",
        "One trainee polishes metal.", "A supervisor guides Diana.",
        "An electrician advises Lisa.", "The foreman thanks Nadia.",
        "Open windows cool the garage.", "Daylight reaches the floor.",
        "Workers carry the damaged panel.", "A customer examines the receipt.",
        "Two technicians inspect wiring.", "The delivery truck arrives.",
        "Neighbors collect restored bicycles.",
    )),
    Scene("garden", "Independent observations during a community garden session.", (
        "Some gardeners dig.", "An observer waits.", "Several children plant seeds.",
        "Diana waters the roses.", "Lisa fills baskets.", "Nadia counts seedlings.",
        "A teacher encourages Diana.", "One volunteer assists Lisa.",
        "The organizer greets Nadia.", "Warm sunlight dries the path.",
        "Small birds gather near the gate.", "New leaves cover the branches.",
        "Parents share fresh bread.", "Two workers mend the fence.",
        "An apprentice carries compost.", "A neighbor prunes the hedge.",
        "Flowers attract bees.", "Rain clouds cross the sky.",
        "Clean boots rest beside the door.", "Visitors enjoy the afternoon.",
    )),
)


def cancel(left: str, reversed_right: str) -> tuple[str, int] | None:
    """Cancel exact leading characters; owner +1 is left, -1 is right."""
    size = min(len(left), len(reversed_right))
    if left[:size] != reversed_right[:size]:
        return None
    return (left[size:], 1) if len(left) >= len(reversed_right) else (reversed_right[size:], -1)


def solve(clauses: tuple[str, ...], *, min_letters: int, max_letters: int,
          max_clauses: int, state_budget: int) -> tuple[list[tuple[int, ...]], dict]:
    """Explore ordered subsets using a nonzero inter-sentence residual."""
    tapes = tuple(letters(text) for text in clauses)
    results = set()
    stats = Counter()
    frontier = [((), (), "", 1, 0, frozenset())]
    while frontier and stats["states"] < state_budget:
        left, right, debt, owner, length, used = frontier.pop()
        stats["states"] += 1
        if used and (not debt or debt == debt[::-1]):
            order = left + right
            target = "".join(tapes[i] for i in order)
            if min_letters <= length <= max_letters and target == target[::-1]:
                results.add(order)
            # Empty debt would force a proper palindrome island if more
            # multiword sentences were inserted inside this closed boundary.
            if not debt:
                stats["closed_boundary_stops"] += 1
                continue
        if len(used) >= max_clauses:
            stats["clause_limit_stops"] += 1
            continue
        for i in reversed(range(len(clauses))):
            if i in used or length + len(tapes[i]) > max_letters:
                continue
            if not used:
                next_state = ((i,), (), tapes[i], 1, len(tapes[i]), frozenset({i}))
            elif owner == 1:
                outcome = cancel(debt, tapes[i][::-1])
                if outcome is None:
                    stats["residual_mismatches"] += 1
                    continue
                residual, next_owner = outcome
                next_state = (left, (i,) + right, residual, next_owner,
                              length + len(tapes[i]), used | {i})
            else:
                outcome = cancel(tapes[i], debt)
                if outcome is None:
                    stats["residual_mismatches"] += 1
                    continue
                residual, next_owner = outcome
                next_state = (left + (i,), right, residual, next_owner,
                              length + len(tapes[i]), used | {i})
            stats["max_sentences_in_state"] = max(stats["max_sentences_in_state"], len(next_state[-1]))
            stats["residual_transitions"] += 1
            frontier.append(next_state)
    return sorted(results), dict(stats) | {"budget_exhausted": bool(frontier),
                                         "unexpanded_states": len(frontier)}


def audit_order(scene: Scene, order: tuple[int, ...], kind: str) -> dict:
    text = " ".join(scene.clauses[i] for i in order)
    audit = independent_audit(text)
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=200)
    checks["independent_exact"] = audit["exact"]
    checks["distinct_source_sentences"] = len(order) == len(set(order))
    return {"scene": scene.identifier, "kind": kind, "source_sentence_indices": order,
            "text": text, "independent_audit": audit, "checks": checks,
            "rejection_codes": [key for key, value in checks.items() if not value],
            "provenance": "Ordered subset of the frozen same-scene authored complete sentences.",
            "reader_evidence": "none; source sentence integrity is a construction constraint, not a reader result"}


def run() -> dict:
    records, searches = [], []
    rng = random.Random(20260914)
    for scene in SCENES:
        found, stats = solve(scene.clauses, min_letters=39, max_letters=200,
                             max_clauses=8, state_budget=100000)
        searches.append({"scene": scene.identifier, "sentences": len(scene.clauses), **stats,
                         "exact_orders": len(found)})
        records.extend(audit_order(scene, order, "exact_word_equation_closure") for order in found)
        # Complete ordinary rendered controls, ranked solely by exact character
        # mismatch count. This diagnostic does not guide or prune exact search.
        controls = set()
        for _ in range(2000):
            order = tuple(rng.sample(range(len(scene.clauses)), rng.choice((4, 5, 6))))
            n = sum(len(letters(scene.clauses[i])) for i in order)
            if 100 <= n <= 160:
                controls.add(order)
        ordered = sorted(controls, key=lambda order: (
            independent_audit(" ".join(scene.clauses[i] for i in order))["mismatched_pair_count"], order))
        records.extend(audit_order(scene, order, "complete_near_miss_control") for order in ordered[:5])
    provenance = {"scenes": [scene.__dict__ for scene in SCENES],
                  "catalogue_material": "none; shared catalogue used only to exclude results",
                  "program_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    return {"status": "complete_whole_discourse_word_equation_experiment",
            "config": {"min_letters": 39, "max_letters": 200, "long_control_floor": 100,
                       "max_sentences": 8, "state_budget_per_scene": 100000,
                       "control_sampling_seed": 20260914, "control_draws_per_scene": 2000},
            "provenance": provenance,
            "provenance_sha256": sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
            "searches": searches, "records": records,
            "exact_closures": sum(row["kind"] == "exact_word_equation_closure" for row in records),
            "next_operator_if_empty": "Compile alternative grammatical constituent realizations at the first conflicting paired clauses while retaining sentence identities and the same multi-sentence residual equation; resume only after a nonempty residual transition is witnessed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite existing output")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "exact_closures": result["exact_closures"],
                      "rendered_records": len(result["records"]), "searches": result["searches"]}, indent=2))


if __name__ == "__main__":
    main()
