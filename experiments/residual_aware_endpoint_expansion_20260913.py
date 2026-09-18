"""Residual-aware endpoint expansion for imperative purpose/location clauses.

Typed action/object/location cores are assembled with several independently
licensed purpose attachments.  A forward opening index and a reverse suffix
index (including every partial-word depth) first find three matching letters;
the scheduler then carries the actual unmatched residual and asks the index
for a semantically compatible attachment extending it.  Only paths reaching
six real outer pairs may expand the interior grammar.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS, MAX_LETTERS = 30, 160
WORD_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class Core:
    identifier: str
    action: str
    object: str
    object_adj: str
    place: str
    place_adj: str
    purpose: str
    purpose_object: str
    purpose_object_adj: str

    @property
    def words(self) -> tuple[str, ...]:
        return (self.action, article(self.object_adj), self.object_adj, self.object,
                "in", article(self.place_adj), self.place_adj, self.place,
                "to", self.purpose, article(self.purpose_object_adj),
                self.purpose_object_adj, self.purpose_object)


def article(adjective: str) -> str:
    return "an" if adjective[:1] in "aeiou" else "a"


# Broad modular role inventory.  The endpoint index discovers compatible
# signatures; no action/object pair is selected from a known palindrome.
ACTION_OBJECTS = {"draft": ("letter", "report"), "edit": ("letter", "report"),
                  "file": ("letter", "report"), "make": ("model", "canvas")}
OBJECT_ADJECTIVES = ("clean", "detailed", "red", "solid", "useful")
PLACES = ("garden", "office", "studio", "workshop")
PLACE_ADJECTIVES = ("quiet", "public", "remote", "small")
PURPOSE_OPTIONS = {
    ("draft", "letter"): (("make", ("card", "note")), ("write", ("guide",))),
    ("draft", "report"): (("share", ("guide", "note")), ("file", ("report",))),
    ("edit", "letter"): (("send", ("note",)), ("write", ("guide",))),
    ("edit", "report"): (("write", ("guide",)), ("share", ("note",)), ("track", ("tide",))),
    ("file", "letter"): (("send", ("note",)),),
    ("file", "report"): (("share", ("guide",)),),
    ("make", "model"): (("show", ("model",)),),
    ("make", "canvas"): (("show", ("canvas",)),),
}
PURPOSE_OBJECT_ADJECTIVES = ("annual", "clean", "extra", "final", "useful")


def make_cores() -> tuple[Core, ...]:
    rows = []
    for action, objects in ACTION_OBJECTS.items():
        for obj in objects:
            for purpose, purpose_objects in PURPOSE_OPTIONS[(action, obj)]:
                for purpose_object in purpose_objects:
                    for obj_adj, place, place_adj, purpose_adj in itertools.product(
                            OBJECT_ADJECTIVES, PLACES, PLACE_ADJECTIVES, PURPOSE_OBJECT_ADJECTIVES):
                        rows.append(Core(
                            f"{action}-{obj}-{purpose}-{purpose_object}-{obj_adj}-{place}-{place_adj}-{purpose_adj}",
                            action, obj, obj_adj, place, place_adj, purpose, purpose_object, purpose_adj))
    return tuple(rows)


CORES = make_cores()


def suffix_rows(core: Core) -> tuple[dict, ...]:
    words = core.words; tail = words[-4:]
    tape = "".join(tail)
    return ({"core": core.identifier, "action": core.action, "words": list(tail),
             "letters": tape, "reversed_letters": tape[::-1]},)


def endpoint_indices(cores: tuple[Core, ...] = CORES) -> dict:
    opening_index, reverse_suffix_index = defaultdict(list), defaultdict(list)
    for core in cores:
        opening = "".join(core.words[:3])
        opening_index[(core.action, opening[:3])].append({"core": core.identifier, "letters": opening})
        for row in suffix_rows(core):
            for depth in range(1, len(row["reversed_letters"]) + 1):
                reverse_suffix_index[(core.action, depth, row["reversed_letters"][:depth])].append(row)
    joins = []
    for core in cores:
        opening = "".join(core.words[:3])
        first = reverse_suffix_index.get((core.action, 3, opening[:3]), ())
        for row in first:
            if row["core"] != core.identifier: continue
            joins.append({"core": core.identifier, "action": core.action, "opening_words": list(core.words[:3]), "opening_prefix": opening[:3],
                          "terminal_words": row["words"], "reverse_terminal_prefix": row["reversed_letters"][:3],
                          "matched_pairs": 3, "semantic_roles": ["agent_action", "purpose_object"]})
    return {"opening_index": {f"{a}:{s}": len(v) for (a, s), v in opening_index.items()},
            "reverse_partial_suffix_index": {f"{a}:{n}:{s}": len(v) for (a, n, s), v in reverse_suffix_index.items()},
            "reverse_partial_suffix_entries": {f"{a}:{n}:{s}": v for (a, n, s), v in reverse_suffix_index.items()},
            "minimum_endpoint_pairs": 3, "joins": joins,
            "eligible_core_ids": sorted({j["core"] for j in joins}),
            "join_scope": "within_core_only; no cross-core semantic composition"}


INDEPENDENT_OBJECTS = {"letter": "artifact", "report": "artifact", "model": "artifact", "canvas": "artifact",
                       "card": "artifact", "note": "artifact", "guide": "artifact", "tide": "event"}
INDEPENDENT_ACTION_OBJECTS = {"draft": {"letter", "report"}, "edit": {"letter", "report"},
                              "file": {"letter", "report"}, "make": {"model", "canvas"}}
INDEPENDENT_PURPOSE_OBJECTS = {
    ("draft", "letter", "make"): {"card", "note"}, ("draft", "letter", "write"): {"guide"},
    ("draft", "report", "share"): {"guide", "note"}, ("draft", "report", "file"): {"report"},
    ("edit", "letter", "send"): {"note"}, ("edit", "letter", "write"): {"guide"},
    ("edit", "report", "write"): {"guide"}, ("edit", "report", "share"): {"note"},
    ("edit", "report", "track"): {"tide"},
    ("file", "letter", "send"): {"note"}, ("file", "report", "share"): {"guide"},
    ("make", "model", "show"): {"model"}, ("make", "canvas", "show"): {"canvas"},
}
INDEPENDENT_OBJECT_ADJECTIVES = frozenset(OBJECT_ADJECTIVES)
INDEPENDENT_PLACE_ADJECTIVES = frozenset(PLACE_ADJECTIVES)
INDEPENDENT_PURPOSE_ADJECTIVES = frozenset(PURPOSE_OBJECT_ADJECTIVES)
INDEPENDENT_PLACES = frozenset(PLACES)


def independent_parse(text: str) -> dict:
    tokens = tuple(WORD_RE.findall(text.lower()))
    if len(tokens) != 13: return {"ok": False, "reason": "wrong_role_arity", "tokens": list(tokens)}
    action, od, oa, obj, prep, pd, pa, place, to, purpose, pod, poa, pobj = tokens
    article_ok = od == article(oa) and pd == article(pa) and pod == article(poa)
    role_ok = (action in INDEPENDENT_ACTION_OBJECTS and obj in INDEPENDENT_ACTION_OBJECTS[action]
               and obj in INDEPENDENT_OBJECTS and oa in INDEPENDENT_OBJECT_ADJECTIVES
               and prep == "in" and pd in {"a", "an", "the"} and pa in INDEPENDENT_PLACE_ADJECTIVES
               and place in INDEPENDENT_PLACES and to == "to" and poa in INDEPENDENT_PURPOSE_ADJECTIVES
               and pobj in INDEPENDENT_OBJECTS
               and pobj in INDEPENDENT_PURPOSE_OBJECTS.get((action, obj, purpose), set()))
    return {"ok": article_ok and role_ok, "agreement_ok": article_ok,
            "valency_ok": role_ok, "tokens": list(tokens), "independent_inventory": True}


def outside_in_ledger(text: str, minimum_pairs: int = 3) -> dict:
    tape = normalize_letters(text); events = []
    for i in range(len(tape) // 2):
        right = len(tape)-1-i
        event = {"pair": i+1, "left_index": i, "right_index": right,
                 "left": tape[i], "right": tape[right], "equal": tape[i] == tape[right]}
        events.append(event)
        if not event["equal"]:
            return {"exact": False, "letters": len(tape), "events": events, "first_mismatch": event,
                    "minimum_pairs_reached": len(events) >= minimum_pairs,
                    "normalized_sha256": sha256(tape.encode()).hexdigest()}
    return {"exact": bool(tape), "letters": len(tape), "events": events, "first_mismatch": None,
            "minimum_pairs_reached": len(events) >= minimum_pairs,
            "normalized_sha256": sha256(tape.encode()).hexdigest()}


def render(words):
    text = " ".join(words); return text[:1].upper() + text[1:] + "."


def schedule_core(core: Core, algebra: dict) -> dict:
    opening = "".join(core.words[:3]); suffix = "".join(core.words[-4:])[::-1]
    depth = 0; residual = ""; ledger = []
    for expected, actual in zip(opening, suffix):
        if expected != actual:
            residual = suffix[depth:]
            break
        depth += 1
        key = f"{core.action}:{depth}:{opening[:depth]}"
        available = [row for row in algebra["reverse_partial_suffix_entries"].get(key, ()) if row["core"] == core.identifier]
        ledger.append({"depth": depth, "opening_char": expected, "terminal_char": actual,
                       "residual_after": suffix[depth:], "source": "reverse_partial_suffix_index",
                       "attachment_options": len(available), "selected_terminal_words": available[0]["words"] if available else []})
    extension = {"attempted_at_depth": depth, "required_depth": 6,
                 "residual_before_extension": residual,
                 "available_next_signature": (opening[depth] if depth < len(opening) else None),
                 "extended": depth >= 6,
                 "inner_expansion_pairs": max(0, len(normalize_letters("".join(core.words))) // 2 - depth) if depth >= 6 else 0}
    text = render(core.words); character_ledger = outside_in_ledger(text, minimum_pairs=3)
    parsed = independent_parse(text); central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [k for k, v in central.items() if not v]
    if not character_ledger["exact"]: codes.append("outside_in_ledger_mismatch")
    if not parsed["ok"]: codes.append("independent_complete_reparse_failed")
    return {"record_kind": "residual_aware_endpoint_schedule", "core": core.identifier, "rendered": text,
            "endpoint_depth": depth, "residual_expansion": extension, "endpoint_pair_ledger": ledger,
            "outside_in_ledger": character_ledger,
            "inner_expansion_ledger": character_ledger["events"][depth:] if depth >= 6 else [],
            "independent_parse": parsed,
            "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes, "reader_status": "human-unreviewed; programmatic checks do not certify readability"}


def run(max_paths_per_channel: int = 100_000) -> dict:
    algebra = endpoint_indices(); eligible = set(algebra["eligible_core_ids"])
    by_channel = defaultdict(list)
    for core in CORES:
        if core.identifier in eligible: by_channel[core.action].append(core)
    records, channels = [], []
    for channel, cores in sorted(by_channel.items()):
        selected = cores[:max_paths_per_channel]
        records.extend(schedule_core(core, algebra) for core in selected)
        channels.append({"channel": channel, "eligible_paths": len(cores), "scheduled_paths": len(selected),
                         "truncated": len(selected) < len(cores), "deepest_endpoint_depth": max((r["endpoint_depth"] for r in records if r["core"].startswith(channel + "-")), default=0)})
    admitted = [r for r in records if r["mechanically_admitted"]]
    return {"status": "residual_aware_endpoint_expansion", "config": {
        "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_paths_per_channel": max_paths_per_channel,
        "partial_word_suffix_index": True, "residual_aware_expansion": True,
        "minimum_outer_pairs_before_inner_expansion": 6, "python_owns_outside_in_ledger": True,
        "independent_parser_inventory": True, "single_intact_imperative_purpose_location_clause": True,
        "corpus_generation": False, "human_readability_required_after_admission": True},
        "authored_core_count": len(CORES), "eligible_core_count": len(eligible),
        "endpoint_algebra": algebra, "channel_coverage": channels, "records": records,
        "exact_candidates": [r for r in records if r["outside_in_ledger"]["exact"]],
        "admitted_candidates": admitted,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "material": "authored modular imperative role and attachment inventory; no catalogue or fixed output",
                       "known_catalogue_check": "central admission gates applied"},
        "next_construction_operator": "Add a semantically licensed attachment family whose indexed reverse partial-word suffix extends the recorded residual beyond six pairs; do not patch a token or reuse a failed endpoint.",
        "reader_facing_next_test": "Only an admitted exact surface may enter randomized blinded intact-prose versus shuffled-control reading; programmatic checks do not certify readability."}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out", required=True, type=Path); parser.add_argument("--max-paths-per-channel", type=int, default=100_000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True); result = run(args.max_paths_per_channel); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "authored": result["authored_core_count"], "eligible": result["eligible_core_count"], "exact": len(result["exact_candidates"]), "admitted": len(result["admitted_candidates"])}, indent=2))


if __name__ == "__main__": main()
