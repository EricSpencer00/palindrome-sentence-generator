"""Residual-driven exact search over complete typed sentence plans.

The earlier clause-product search required the two sides of the midpoint to
be complete clauses.  This constructor removes that artificial boundary: a
single complete sentence is compiled as typed lexical slots, and the product
kernel grows it from both ends while carrying the unmatched character
residual.  The midpoint may fall inside any word.  Corpus frequencies are
proposal ordering only; no model is queried for a candidate-level judgment.

Every closure is independently re-normalized and re-audited.  Passing this
file's checks is mechanical eligibility only; readability still requires the
blinded human study package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.whole_text_palindrome_product_20260913 import (
    Grammar,
    compile_slots,
    construct,
)
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


@dataclass(frozen=True)
class Plan:
    name: str
    relation: str
    roles: tuple[str, ...]
    slots: tuple[tuple[str, ...], ...]


DET_SING = ("a", "an", "the", "my", "our", "his", "her", "this", "that")
DET_PLUR = ("the", "some", "these", "those", "our", "their")
NOUN_SING_PERSON = (
    "aide", "artist", "baker", "child", "cook", "doctor", "farmer", "friend",
    "gardener", "guard", "helper", "man", "parent", "poet", "pupil", "teacher",
    "worker", "writer", "reader", "singer", "dancer", "pilot", "editor",
)
NOUN_PLUR_PERSON = (
    "artists", "bakers", "children", "cooks", "doctors", "farmers", "friends",
    "gardeners", "guards", "helpers", "men", "parents", "poets", "pupils",
    "teachers", "workers", "writers", "readers", "singers", "dancers", "pilots",
    "editors",
)
NOUN_SING_THING = (
    "note", "book", "map", "letter", "memo", "plan", "key", "door", "song",
    "story", "task", "test", "cup", "cake", "room", "garden", "bread", "tool",
    "horse", "dog", "cat", "bird", "gate", "road", "river", "flower", "apple",
    "drawer", "star", "time", "diary", "school", "town", "night", "rain", "wind",
    "fire", "water", "light", "mail", "message", "table", "chair", "house", "home",
    "car", "card", "money", "truth", "music", "place", "life", "work", "hand",
    "part", "age", "idea", "issue", "reason", "cause", "result", "event", "film",
    "team", "group", "name", "answer", "code", "data", "line", "word",
)
NOUN_PLUR_THING = (
    "notes", "books", "maps", "letters", "memos", "plans", "keys", "doors", "songs",
    "stories", "tasks", "tests", "cups", "cakes", "rooms", "gardens", "tools",
    "horses", "dogs", "cats", "birds", "gates", "roads", "rivers", "flowers", "apples",
    "drawers", "stars", "times", "diaries", "schools", "towns", "nights", "rains",
    "winds", "fires", "waters", "lights", "mails", "messages", "tables", "chairs",
    "houses", "homes", "cars", "cards", "people", "truths", "places", "lives", "works",
    "hands", "parts", "ages", "ideas", "issues", "reasons", "causes", "results", "events",
    "films", "teams", "groups", "names", "answers", "codes", "lines", "words",
)
VERB_PAST = (
    "aided", "asked", "baked", "built", "called", "carried", "changed", "cleaned",
    "closed", "cooked", "drew", "drove", "ate", "found", "fixed", "helped", "held",
    "kept", "learned", "liked", "loved", "made", "marked", "met", "moved", "noticed",
    "opened", "painted", "planned", "read", "repaired", "rescued", "saved", "saw", "sent",
    "shared", "showed", "studied", "taught", "thanked", "told", "used", "visited", "watched",
    "wrote", "ripped", "inspired", "served", "recorded", "tested", "worked", "mailed",
    "rewarded", "delivered", "repaid", "stressed", "started", "ended", "stated", "got",
    "gave", "took", "put", "brought", "left", "meant", "needed", "knew",
)
VERB_BASE = (
    "aid", "ask", "bake", "build", "call", "carry", "change", "clean", "close", "cook",
    "draw", "drive", "eat", "find", "fix", "help", "hold", "keep", "learn", "like", "love",
    "make", "mark", "meet", "move", "notice", "open", "paint", "plan", "read", "repair",
    "rescue", "save", "see", "send", "share", "show", "study", "teach", "thank", "tell", "use",
    "visit", "watch", "write", "rip", "inspire", "serve", "record", "test", "work", "mail",
    "reward", "deliver", "repay", "stress", "start", "end", "state", "get", "give", "take",
    "put", "bring", "leave", "need", "know",
)
ADJ = (
    "calm", "careful", "kind", "quiet", "small", "bright", "dark", "clear", "open", "warm",
    "cold", "good", "true", "old", "young", "patient", "useful", "brave", "new", "long",
    "short", "safe", "gentle", "ready", "full", "empty", "red", "green",
)
PREP = ("at", "in", "on", "by", "for", "with", "from", "near", "after", "before", "under", "over", "into", "through", "around")
NAMES = (
    "diana", "anna", "emma", "grace", "helen", "jane", "laura", "maria", "maya", "nina",
    "sara", "sophia", "liam", "delia", "noel", "leon", "emil", "damon", "evan", "alice",
    "amelia", "amy", "aria", "ava", "bella", "carla", "clara", "ella", "elena", "elise",
    "emily", "erica", "faith", "fiona", "hannah", "iris", "isabel", "jade", "julia", "kara",
    "lauren", "leah", "lily", "lisa", "lucy", "mia", "naomi", "natalie", "nicole", "nora",
    "olivia", "paige", "rachel", "rebecca", "rose", "ruby", "samantha", "sharon", "stella",
    "susan", "tanya", "teresa", "tina", "valerie", "victoria", "viola", "vivian", "wendy", "zoe",
)
AUX = ("is", "was", "are", "were", "am")


def _unique(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(word for group in groups for word in group))


def plans() -> tuple[Plan, ...]:
    """Six relation families with three complete, independently typed forms."""
    s = []
    # SVO variants.  Singular and plural determiner/noun pairings are kept in
    # separate plans so a complete closure can be reparsed without inference.
    s.extend([
        Plan("svo_past_sing", "artifact_transfer",
             ("subject_det", "subject_adj", "subject", "verb", "object_det", "object_adj", "object"),
             (DET_SING, ADJ, NOUN_SING_PERSON, VERB_PAST, DET_SING, ADJ, NOUN_SING_THING)),
        Plan("svo_past_plural", "artifact_transfer",
             ("subject_det", "subject_adj", "subject", "verb", "object_det", "object_adj", "object"),
             (DET_PLUR, ADJ, NOUN_PLUR_PERSON, VERB_PAST, DET_PLUR, ADJ, NOUN_PLUR_THING)),
        Plan("svo_base_sing", "artifact_transfer",
             ("subject_det", "subject_adj", "subject", "verb", "object_det", "object_adj", "object"),
             (DET_SING, ADJ, NOUN_SING_PERSON, VERB_BASE, DET_SING, ADJ, NOUN_SING_THING)),
        Plan("svo_pp_sing", "located_action",
             ("subject_det", "subject", "verb", "object_det", "object", "prep", "place_det", "place"),
             (DET_SING, NOUN_SING_PERSON, VERB_PAST, DET_SING, NOUN_SING_THING, PREP, DET_SING, NOUN_SING_THING)),
        Plan("svo_pp_plural", "located_action",
             ("subject_det", "subject", "verb", "object_det", "object", "prep", "place_det", "place"),
             (DET_PLUR, NOUN_PLUR_PERSON, VERB_PAST, DET_PLUR, NOUN_PLUR_THING, PREP, DET_PLUR, NOUN_PLUR_THING)),
        Plan("named_transfer", "named_event",
             ("name", "verb", "object_det", "object_adj", "object"),
             (NAMES, VERB_PAST, DET_SING, ADJ, NOUN_SING_THING)),
        Plan("copular_sing", "property",
             ("subject_det", "subject_adj", "subject", "aux", "complement_adj"),
             (DET_SING, ADJ, NOUN_SING_PERSON, AUX, ADJ)),
        Plan("copular_plural", "property",
             ("subject_det", "subject_adj", "subject", "aux", "complement_adj"),
             (DET_PLUR, ADJ, NOUN_PLUR_PERSON, AUX, ADJ)),
        Plan("imperative", "instruction",
             ("verb", "object_det", "object_adj", "object", "prep", "place_det", "place"),
             (VERB_BASE, DET_SING, ADJ, NOUN_SING_THING, PREP, DET_SING, NOUN_SING_THING)),
        Plan("intransitive", "event",
             ("subject_det", "subject_adj", "subject", "verb"),
             (DET_SING, ADJ, NOUN_SING_PERSON, VERB_PAST)),
        Plan("named_property", "named_property",
             ("name", "aux", "complement_adj", "prep", "place_det", "place"),
             (NAMES, AUX, ADJ, PREP, DET_SING, NOUN_SING_THING)),
        Plan("question_like", "request",
             ("verb", "subject_det", "subject", "object_det", "object"),
             (VERB_BASE, DET_SING, NOUN_SING_PERSON, DET_SING, NOUN_SING_THING)),
    ])
    return tuple(s)


def _reparse(plan: Plan, words: tuple[str, ...]) -> dict[str, object]:
    if len(words) != len(plan.slots):
        return {"ok": False, "reason": "word_count", "tokens": list(words)}
    for word, choices, role in zip(words, plan.slots, plan.roles):
        if word not in choices:
            return {"ok": False, "reason": f"unknown_{role}", "tokens": list(words)}
    return {"ok": True, "roles": list(plan.roles), "tokens": list(words)}


def _audit(plan: Plan, words: tuple[str, ...], kernel: dict[str, object]) -> dict[str, object]:
    rendered = " ".join(words)
    tape = normalize_letters(rendered)
    independent = "".join(ch.lower() for ch in rendered if ch.isascii() and ch.isalpha())
    checks = mechanical_admission_checks(rendered, min_letters=39, max_letters=140)
    checks["independent_exact_audit"] = bool(tape) and tape == tape[::-1] and tape == independent
    parse = _reparse(plan, words)
    checks["independent_complete_reparse"] = bool(parse["ok"])
    return {
        "plan": plan.name,
        "relation": plan.relation,
        "roles": list(plan.roles),
        "rendered": rendered,
        "words": list(words),
        "letters": len(tape),
        "independent_normalized_letters": independent,
        "independent_exact_audit": checks["independent_exact_audit"],
        "independent_reparse": parse,
        "kernel": kernel,
        "mechanical_checks": checks,
        "mechanically_eligible": all(checks.values()),
        "reader_status": "human-unreviewed; diagnostics never certify readability",
    }


def run(*, max_states: int = 100_000) -> dict[str, object]:
    if max_states < 1:
        raise ValueError("max_states must be positive")
    records: list[dict[str, object]] = []
    plan_runs: list[dict[str, object]] = []
    for plan in plans():
        grammar: Grammar = compile_slots(plan.slots)
        result = construct(grammar, max_states=max_states)
        rows = []
        for item in result["records"]:
            row = _audit(plan, tuple(item["words"]), {
                "states": result["states"],
                "center_characters": item["center_characters"],
                "midpoint_letter_offset": item["midpoint_letter_offset"],
                "kernel_exact": item["exact"],
            })
            rows.append(row)
            records.append(row)
        plan_runs.append({
            "plan": plan.name,
            "relation": plan.relation,
            "roles": list(plan.roles),
            "slot_sizes": [len(slot) for slot in plan.slots],
            "states": result["states"],
            "pending_states": result["pending_states"],
            "truncated": result["truncated"],
            "closures": len(rows),
            "mechanically_eligible": sum(bool(row["mechanically_eligible"]) for row in rows),
        })
    return {
        "status": "residual_typed_complete_sentence_search",
        "config": {
            "max_states_per_plan": max_states,
            "minimum_letters": 39,
            "maximum_letters": 140,
            "free_midpoint": True,
            "residual_driven_expansion": True,
            "independent_complete_reparse": True,
            "machine_readability_certification": False,
            "no_catalogue_text": True,
            "no_per_candidate_model": True,
        },
        "plan_runs": plan_runs,
        "records": records,
        "mechanically_eligible": [row for row in records if row["mechanically_eligible"]],
        "provenance": {
            "generator": "residual_typed_sentence_search_20260914",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "kernel": "whole_text_palindrome_product_20260913",
            "material": "authored typed lexical inventories; no borrowed sentence spans",
            "construction": "one complete sentence with midpoint allowed inside any word",
        },
        "reader_facing_next_test": "Only an original mechanically eligible closure may enter a randomized blinded intact-prose versus shuffled-control study.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=100_000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(max_states=args.max_states)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(result["records"]),
                      "mechanically_eligible": len(result["mechanically_eligible"])}, sort_keys=True))


if __name__ == "__main__":
    main()
