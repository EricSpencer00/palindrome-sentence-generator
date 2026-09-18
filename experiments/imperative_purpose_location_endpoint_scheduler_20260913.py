"""Endpoint-indexed scheduler for typed imperative purpose/location clauses.

This grammar is disjoint from the earlier declarative routes.  Broad authored
role inventories generate ordinary imperative actions with a locative and a
purpose attachment.  A reverse index over the terminal purpose-object phrase
is joined to a forward index over the opening action before a path is
scheduled.  Only joins with at least three real outer character matches enter
the Python-owned ledger.  The independent parser uses a separate inventory and
frame map, not the generated frame object.
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
class Frame:
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
                "to", self.purpose, article(self.purpose_object_adj), self.purpose_object_adj,
                self.purpose_object)


def article(adjective: str) -> str:
    return "an" if adjective[:1] in "aeiou" else "a"


# Independent authored role inventories.  Compatibility maps below are
# action-specific rather than a generic artifact type.
ACTION_OBJECTS = {
    "repair": ("canvas", "model"),
    "paint": ("canvas",),
    "draft": ("letter", "report"),
    "build": ("model",),
}
OBJECT_ADJECTIVES = ("clean", "damaged", "detailed", "solid")
PLACES = ("garden", "office", "studio", "workshop")
PLACE_ADJECTIVES = ("quiet", "public", "remote", "small")
PURPOSES = {
    ("repair", "canvas"): (("save", ("paper", "report")), ("keep", ("letter",))),
    ("repair", "model"): (("protect", ("report",)), ("save", ("paper",))),
    ("paint", "canvas"): (("display", ("canvas",)), ("frame", ("paper",))),
    ("draft", "letter"): (("send", ("letter",)), ("keep", ("paper",))),
    ("draft", "report"): (("file", ("report",)), ("share", ("paper",))),
    ("build", "model"): (("test", ("model",)), ("show", ("model",))),
}
PURPOSE_OBJECT_ADJECTIVES = ("annual", "final", "useful", "clean")


def frames() -> tuple[Frame, ...]:
    rows = []
    for action, objects in ACTION_OBJECTS.items():
        for obj in objects:
            for purpose, purpose_objects in PURPOSES[(action, obj)]:
                for obj_adj, place, place_adj, purpose_obj_adj, purpose_obj in itertools.product(
                    OBJECT_ADJECTIVES, PLACES, PLACE_ADJECTIVES, PURPOSE_OBJECT_ADJECTIVES, purpose_objects):
                    rows.append(Frame(f"{action}-{obj}-{purpose}-{purpose_obj}-{obj_adj}-{place}-{place_adj}-{purpose_obj_adj}",
                                      action, obj, obj_adj, place, place_adj, purpose, purpose_obj, purpose_obj_adj))
    return tuple(rows)


FRAMES = frames()


def endpoint_index(frameset: tuple[Frame, ...] = FRAMES) -> dict:
    opening, terminal = defaultdict(list), defaultdict(list)
    for frame in frameset:
        words = frame.words
        opening[words[0][:3]].append({"frame": frame.identifier, "words": [words[0]], "letters": words[0]})
        tail = words[-3:]
        tail_letters = "".join(tail)
        terminal[tail_letters[::-1][:3]].append({"frame": frame.identifier, "words": list(tail), "letters": tail_letters})
    joins = []
    for frame in frameset:
        words = frame.words; key = words[0][:3]
        for end in terminal.get(key, ()):
            if end["frame"] != frame.identifier: continue
            opening_letters = words[0]
            reversed_tail = end["letters"][::-1]
            matched = sum(1 for left, right in itertools.takewhile(lambda pair: pair[0] == pair[1], zip(opening_letters, reversed_tail)))
            if matched < 3: continue
            joins.append({"frame": frame.identifier, "opening_words": [words[0],],
                          "terminal_words": end["words"], "opening_letters": opening_letters,
                          "reversed_terminal_letters": reversed_tail, "matched_pairs": matched,
                          "semantic_roles": ["agent_action", "purpose_object"],
                          "chosen_surface_words": list(words)})
    return {"minimum_matched_pairs": 3, "opening_index": {k: len(v) for k, v in opening.items()},
            "reverse_terminal_index": {k: len(v) for k, v in terminal.items()},
            "joins": joins, "eligible_frame_ids": sorted({j["frame"] for j in joins}),
            "join_scope": "within_frame_only; no cross-frame role composition"}


def outside_in_ledger(text: str, minimum_pairs: int = 3) -> dict:
    tape = normalize_letters(text); events = []
    for i in range(len(tape) // 2):
        right = len(tape) - 1 - i
        event = {"pair": i + 1, "left_index": i, "right_index": right,
                 "left": tape[i], "right": tape[right], "equal": tape[i] == tape[right]}
        events.append(event)
        if not event["equal"]:
            return {"exact": False, "letters": len(tape), "events": events,
                    "first_mismatch": event, "minimum_pairs_reached": len(events) >= minimum_pairs,
                    "normalized_sha256": sha256(tape.encode()).hexdigest()}
    return {"exact": bool(tape), "letters": len(tape), "events": events,
            "first_mismatch": None, "minimum_pairs_reached": len(events) >= minimum_pairs,
            "normalized_sha256": sha256(tape.encode()).hexdigest()}


# Separate parser declarations intentionally duplicate the linguistic facts in
# an independent form instead of reading Frame fields or generated choices.
PARSE_ACTION_OBJECTS = {"repair": {"canvas", "model"}, "paint": {"canvas"},
                        "draft": {"letter", "report"}, "build": {"model"}}
PARSE_PURPOSE_OBJECTS = {
    ("repair", "canvas", "save"): {"paper", "report"}, ("repair", "canvas", "keep"): {"letter"},
    ("repair", "model", "protect"): {"report"}, ("repair", "model", "save"): {"paper"},
    ("paint", "canvas", "display"): {"canvas"}, ("paint", "canvas", "frame"): {"paper"},
    ("draft", "letter", "send"): {"letter"}, ("draft", "letter", "keep"): {"paper"},
    ("draft", "report", "file"): {"report"}, ("draft", "report", "share"): {"paper"},
    ("build", "model", "test"): {"model"}, ("build", "model", "show"): {"model"},
}
PARSE_SUBJECT_TYPE = "understood_person_agent"
PARSE_OBJECT_ADJ = frozenset(OBJECT_ADJECTIVES)
PARSE_PLACE_ADJ = frozenset(PLACE_ADJECTIVES)
PARSE_PURPOSE_ADJ = frozenset(PURPOSE_OBJECT_ADJECTIVES)
PARSE_OBJECTS = frozenset({"canvas", "model", "letter", "report"})
PARSE_PLACES = frozenset(PLACES)


def independent_parse(text: str) -> dict:
    tokens = tuple(WORD_RE.findall(text.lower()))
    if len(tokens) != 13: return {"ok": False, "reason": "wrong_role_arity", "tokens": list(tokens)}
    action, od, oa, obj, prep, pd, pa, place, to, purpose, pod, poa, pobj = tokens
    article_ok = od == article(oa) and pd == article(pa) and pod == article(poa)
    role_ok = (action in PARSE_ACTION_OBJECTS and obj in PARSE_ACTION_OBJECTS[action] and oa in PARSE_OBJECT_ADJ
               and prep == "in" and pa in PARSE_PLACE_ADJ and place in PARSE_PLACES and to == "to"
               and (action, obj, purpose) in PARSE_PURPOSE_OBJECTS and pobj in PARSE_PURPOSE_OBJECTS[(action, obj, purpose)]
               and poa in PARSE_PURPOSE_ADJ and od in {"a", "an", "the"} and pd in {"a", "an", "the"} and pod in {"a", "an", "the"})
    return {"ok": article_ok and role_ok, "agreement_ok": article_ok,
            "valency_ok": role_ok, "semantic_subject_type": PARSE_SUBJECT_TYPE,
            "tokens": list(tokens), "independent_inventory": True}


def audit(frame: Frame, join: dict) -> dict:
    text = render(frame.words); ledger = outside_in_ledger(text)
    parsed = independent_parse(text); central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in central.items() if not value]
    if not ledger["exact"]: codes.append("outside_in_ledger_mismatch")
    if not parsed["ok"]: codes.append("independent_complete_reparse_failed")
    return {"record_kind": "imperative_purpose_location_endpoint_schedule", "frame": frame.identifier,
            "rendered": text, "endpoint_join": join, "outside_in_ledger": ledger,
            "independent_parse": parsed, "central_admission": central,
            "mechanically_admitted": not codes, "rejection_codes": codes,
            "reader_status": "human-unreviewed; programmatic checks do not certify readability"}


def run(max_paths_per_channel: int = 100_000) -> dict:
    algebra = endpoint_index(); joins_by = defaultdict(list)
    for join in algebra["joins"]: joins_by[join["opening_letters"][:3]].append(join)
    by_id = {frame.identifier: frame for frame in FRAMES}
    records, channels = [], []
    for channel, joins in sorted(joins_by.items()):
        scheduled = joins[:max_paths_per_channel]
        for join in scheduled: records.append(audit(by_id[join["frame"]], join))
        channels.append({"channel": channel, "eligible_paths": len(joins), "scheduled_paths": len(scheduled),
                         "truncated": len(scheduled) < len(joins)})
    admitted = [row for row in records if row["mechanically_admitted"]]
    return {"status": "imperative_purpose_location_endpoint_scheduler", "config": {
        "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_paths_per_channel": max_paths_per_channel,
        "endpoint_index_before_ledger": True, "minimum_endpoint_matched_pairs": 3,
        "python_owns_outside_in_ledger": True, "independent_complete_reparse": True,
        "independent_parser_inventory": True, "single_intact_imperative_clause": True,
        "corpus_generation": False, "human_readability_required_after_admission": True},
        "authored_frame_count": len(FRAMES), "eligible_frame_count": len(algebra["eligible_frame_ids"]),
        "endpoint_algebra": algebra, "channel_coverage": channels, "records": records,
        "exact_candidates": [row for row in records if row["outside_in_ledger"]["exact"]],
        "admitted_candidates": admitted,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "material": "authored frame-specific imperative purpose/location roles; no catalogue or fixed output",
                       "known_catalogue_check": "central admission gates applied after independent ledger/parser"},
        "next_construction_operator": "Add a new modular purpose/object role family, rerun endpoint indexing, and preserve per-channel ledger coverage; do not patch a mismatching word.",
        "reader_facing_next_test": "Only an admitted exact surface may enter randomized blinded intact-prose versus shuffled-control reading; programmatic checks do not certify readability."}


def render(words):
    text = " ".join(words); return text[:1].upper() + text[1:] + "."


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out", required=True, type=Path); parser.add_argument("--max-paths-per-channel", type=int, default=100_000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True); result = run(args.max_paths_per_channel); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "authored": result["authored_frame_count"], "eligible": result["eligible_frame_count"], "exact": len(result["exact_candidates"]), "admitted": len(result["admitted_candidates"])}, indent=2))


if __name__ == "__main__": main()
