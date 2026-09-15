"""Exact residual search over two independently typed clause plans.

Words are emitted from the left edge of one clause and the right edge of the
other.  Their actual letters cancel immediately, so word boundaries may be
staggered without delaying syntax until ranking.  Every plan is a complete
agreement/valency-safe clause; exact outputs still need blinded readers.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


@dataclass(frozen=True)
class Plan:
    name: str
    roles: tuple[str, ...]
    pools: tuple[tuple[str, ...], ...]


DET = ("a", "an", "the", "this", "that", "my", "our")
PERSON = ("aide", "artist", "author", "baker", "doctor", "editor", "farmer",
          "friend", "gardener", "nurse", "pilot", "sailor", "teacher", "writer")
PERSON_PL = ("aides", "artists", "authors", "bakers", "doctors", "editors",
             "farmers", "friends", "gardeners", "men", "nurses", "pilots",
             "sailors", "teachers", "writers")
THING = ("book", "bread", "canvas", "door", "gift", "letter", "map", "memo",
         "note", "plan", "report", "road", "room", "story", "tray")
THING_PL = ("books", "doors", "gifts", "letters", "maps", "memos", "notes",
            "plans", "reports", "roads", "rooms", "stories", "trays")
NAME = ("Aidan", "Alan", "Ari", "Diana", "Eva", "Ira", "Liam", "Mia",
        "Nadia", "Nora", "Noel")
NUM = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
QUANT = ("some", "many", "several", "two", "three", "nine")
VT_SG = ("admires", "calls", "finds", "helps", "inspires", "keeps", "likes",
         "needs", "opens", "reads", "rips", "saves", "sees", "sends", "writes")
VT_PL = ("admire", "call", "find", "help", "inspire", "keep", "like", "need",
         "open", "read", "rip", "save", "see", "send", "write")
VT_PAST = ("admired", "called", "found", "helped", "kept", "liked", "needed",
           "opened", "read", "repaid", "saved", "saw", "sent", "wrote")
BASE = ("admire", "call", "find", "help", "keep", "mail", "open", "read",
        "save", "send", "write")
ADJ_PERSON = ("calm", "careful", "kind", "patient", "quiet", "young")
ADJ_THING = ("brief", "clear", "fresh", "new", "old", "open", "small", "warm")
ADVERB = ("carefully", "early", "gently", "often", "quietly", "today")
PREP = ("by", "for", "in", "near", "on", "to", "with")


PLANS = (
    Plan("numbered_svo", ("det", "person", "vt_sg", "number", "thing_pl"),
         (DET, PERSON, VT_SG, NUM, THING_PL)),
    Plan("plural_name", ("quant", "person_pl", "vt_pl", "name"),
         (QUANT, PERSON_PL, VT_PL, NAME)),
    Plan("singular_svo", ("det", "person", "vt_sg", "det", "thing"),
         (DET, PERSON, VT_SG, DET, THING)),
    Plan("name_svo", ("name", "vt_sg", "det", "thing"),
         (NAME, VT_SG, DET, THING)),
    Plan("past_svo", ("det", "person", "vt_past", "det", "thing"),
         (DET, PERSON, VT_PAST, DET, THING)),
    Plan("adj_svo", ("det", "adj_person", "person", "vt_sg", "det", "adj_thing", "thing"),
         (DET, ADJ_PERSON, PERSON, VT_SG, DET, ADJ_THING, THING)),
    Plan("adv_svo", ("det", "person", "vt_sg", "det", "thing", "adverb"),
         (DET, PERSON, VT_SG, DET, THING, ADVERB)),
    Plan("imperative", ("base_vt", "det", "thing", "prep", "name"),
         (BASE, DET, THING, PREP, NAME)),
    Plan("relative_singular",
         ("det", "person", "relative", "vt_sg", "det", "thing", "vt_sg", "det", "thing"),
         (DET, PERSON, ("who",), VT_SG, DET, THING, VT_SG, DET, THING)),
    Plan("relative_plural_name",
         ("quant", "person_pl", "relative", "vt_pl", "det", "thing", "vt_pl", "name"),
         (QUANT, PERSON_PL, ("who",), VT_PL, DET, THING, VT_PL, NAME)),
    Plan("coordinated_singular",
         ("det", "person", "vt_sg", "det", "thing", "coord", "vt_sg", "det", "thing"),
         (DET, PERSON, VT_SG, DET, THING, ("and",), VT_SG, DET, THING)),
    Plan("coordinated_plural_name",
         ("quant", "person_pl", "vt_pl", "name", "coord", "vt_pl", "det", "thing"),
         (QUANT, PERSON_PL, VT_PL, NAME, ("and",), VT_PL, DET, THING)),
)


def ascii_audit(text: str) -> dict:
    tape = "".join(char.casefold() for char in text if char.isascii() and char.isalpha())
    mismatches = [[i, len(tape) - 1 - i] for i in range(len(tape) // 2)
                  if tape[i] != tape[-1 - i]]
    return {"normalized": tape, "letters": len(tape), "mismatches": mismatches,
            "exact": bool(tape) and not mismatches}


def cancel(residual: str, emitted: str, owner: int) -> tuple[str, int] | None:
    """Cancel two streams in their shared outside-to-inside orientation."""
    if owner not in (-1, 1):
        raise ValueError("owner must be -1 or 1")
    common = min(len(residual), len(emitted))
    if residual[:common] != emitted[:common]:
        return None
    if len(residual) > common:
        return residual[common:], owner
    if len(emitted) > common:
        return emitted[common:], -owner
    return "", 0


def search_pair(left: Plan, right: Plan, *, state_budget: int = 250_000,
                result_limit: int = 200) -> tuple[list[dict], dict]:
    results: list[dict] = []
    states = 0
    deepest = {"matched_letters": 0, "left_words": (), "right_words": (),
               "residual": "", "owner": 0}
    # li advances through the visible left clause. ri moves backwards through
    # the visible right clause because reverse(right) is consumed from outside.
    stack = [(0, len(right.roles) - 1, "", 0, (), (), 0)]
    while stack and states < state_budget and len(results) < result_limit:
        li, ri, residual, owner, left_words, right_reversed, matched = stack.pop()
        states += 1
        if matched > deepest["matched_letters"]:
            deepest = {"matched_letters": matched, "left_words": left_words,
                       "right_words": tuple(reversed(right_reversed)),
                       "residual": residual, "owner": owner}
        if li == len(left.roles) and ri < 0:
            if not residual:
                ltext = " ".join(left_words)
                rtext = " ".join(reversed(right_reversed))
                results.append({"text": f"{ltext.capitalize()}; {rtext}.",
                                "left_plan": left.name, "right_plan": right.name,
                                "left_words": left_words,
                                "right_words": tuple(reversed(right_reversed))})
            continue
        if owner == 0:
            if li >= len(left.roles):
                continue
            for word in reversed(left.pools[li]):
                tape = normalize_letters(word)
                stack.append((li + 1, ri, tape, 1, left_words + (word,),
                              right_reversed, matched))
        elif owner == 1:
            if ri < 0:
                continue
            for word in reversed(right.pools[ri]):
                emitted = normalize_letters(word)[::-1]
                outcome = cancel(residual, emitted, 1)
                if outcome is None:
                    continue
                new_residual, new_owner = outcome
                stack.append((li, ri - 1, new_residual, new_owner, left_words,
                              right_reversed + (word,), matched + min(len(residual), len(emitted))))
        else:
            if li >= len(left.roles):
                continue
            for word in reversed(left.pools[li]):
                emitted = normalize_letters(word)
                outcome = cancel(residual, emitted, -1)
                if outcome is None:
                    continue
                new_residual, new_owner = outcome
                stack.append((li + 1, ri, new_residual, new_owner,
                              left_words + (word,), right_reversed,
                              matched + min(len(residual), len(emitted))))
    return results, {"states": states, "budget_exhausted": bool(stack), "deepest": deepest}


def run(*, state_budget_per_pair: int = 250_000) -> dict:
    exact, eligible, searches = [], [], []
    for left in PLANS:
        for right in PLANS:
            rows, stats = search_pair(left, right, state_budget=state_budget_per_pair)
            searches.append({"left": left.name, "right": right.name, **stats,
                             "exact_closures": len(rows)})
            for row in rows:
                audit = ascii_audit(row["text"])
                checks = mechanical_admission_checks(row["text"], min_letters=39,
                                                      max_letters=160)
                row = {**row, "audit": audit, "mechanical_checks": checks,
                       "mechanically_eligible": all(checks.values()),
                       "reader_status": "human-unreviewed"}
                exact.append(row)
                if row["mechanically_eligible"]:
                    eligible.append(row)
    unique = {row["audit"]["normalized"]: row for row in exact}
    return {
        "status": "dual_plan_residual_search_complete",
        "config": {"plans": len(PLANS), "state_budget_per_pair": state_budget_per_pair,
                   "min_letters": 39, "grammar_during_search": True,
                   "staggered_word_boundaries": True},
        "exact_closures": len(exact), "unique_exact_closures": len(unique),
        "eligible_closures": eligible, "exact_records": list(unique.values()),
        "searches": searches,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "material": "authored typed lexical products; no catalogue text"},
        "next_operator_if_empty": "Mine additional valency-safe lexical choices by residual prefix, then add relative-clause and coordinated-clause plan states before replay.",
        "reader_next": "Any mechanically eligible novel closure enters randomized blinded intact-prose versus shuffled-control rating; code never certifies readability.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-budget", type=int, default=250_000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite output")
    result = run(state_budget_per_pair=args.state_budget)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("exact_closures", "unique_exact_closures")}, indent=2))


if __name__ == "__main__":
    main()
