"""Typed semantic event-frame CSP with independently selected arguments.

The earlier event lane reused the left setting and generated a large relation
product before checking the tape.  This repair first builds a finite bank of
complete, typed event frames (agreement, valency, and argument classes), then
joins two independently selected frames through a punctuation or relation
seam.  Exact character equations are evaluated only after both sides are
chosen; no finished palindrome is copied to manufacture a hit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


@dataclass(frozen=True)
class Event:
    actor: str
    actor_class: str
    verb: str
    verb_number: str
    obj: str
    obj_class: str
    place: str

    @property
    def text(self) -> str:
        return " ".join(part for part in (self.actor, self.verb, self.obj, self.place) if part)


ACTORS = (
    ("an aide", "animate"), ("some men", "animate_plural"), ("a poet", "animate"),
    ("the editor", "animate"), ("a teacher", "animate"), ("the nurse", "animate"),
    ("Diana", "animate"), ("Noel", "animate"), ("Leon", "animate"),
)
VERBS = (
    ("rips", "SG", "document"), ("reads", "SG", "document"),
    ("writes", "SG", "document"), ("marks", "SG", "document"),
    ("inspires", "SG", "animate"), ("inspire", "PL", "animate"),
    ("saw", "ANY", "any"), ("helps", "SG", "animate"),
    ("help", "PL", "animate"), ("guides", "SG", "animate"),
)
OBJECTS = (
    ("nine memos", "document", "PL"), ("some prose", "document", "MASS"),
    ("a note", "document", "SG"), ("the map", "document", "SG"),
    ("a poem", "document", "SG"), ("the letter", "document", "SG"),
    ("Diana", "animate", "SG"), ("Noel", "animate", "SG"),
    ("Leon", "animate", "SG"), ("some men", "animate", "PL"),
)
PLACES = ("", "at dawn", "in town", "near home", "by the sea", "in an arena")
SEAMS = ((";", "adjacent"), (".", "sentence_boundary"),
         (" because ", "causal"), (" while ", "contrastive"), (" as ", "temporal"))
KNOWN_BASELINE_TAPES = {
    "anaideripsninememossomemeninspirediana",
}


def tape(text: str) -> str:
    return normalize_letters(text)


def audit(text: str) -> dict:
    t = tape(text)
    r = t[::-1]
    f = hashlib.sha256(t.encode()).hexdigest()
    rr = hashlib.sha256(r.encode()).hexdigest()
    bad = [(i, len(t) - 1 - i) for i in range(len(t) // 2) if t[i] != t[-1 - i]]
    return {
        "letters": len(t),
        "two_pointer_exact": not bad,
        "mismatch_count": len(bad),
        "first_mismatch": bad[0] if bad else None,
        "sha256_forward": f,
        "sha256_reverse": rr,
        "sha_equal_under_reversal": f == rr,
    }


def build_events() -> tuple[Event, ...]:
    events: list[Event] = []
    for actor, actor_class in ACTORS:
        number = "PL" if actor_class == "animate_plural" else "SG"
        for verb, verb_number, valency in VERBS:
            if verb_number not in ("ANY", number):
                continue
            for obj, obj_class, _obj_number in OBJECTS:
                if valency not in ("any", obj_class):
                    continue
                for place in PLACES:
                    events.append(Event(actor, actor_class, verb, verb_number, obj, obj_class, place))
    return tuple(events)


def run(*, max_probes: int = 10_000) -> dict:
    events = build_events()
    rows: list[dict] = []
    exact_rows: list[dict] = []
    checked = 0
    # The first version accidentally enumerated the full Cartesian product even
    # when a bounded diagnostic was requested.  Keep the diagnostic genuinely
    # bounded so a failed lane cannot hide behind an unbounded sweep.
    stop = False
    for left in events:
        for seam, seam_name in SEAMS:
            for right in events:
                checked += 1
                rendered = f"{left.text}{seam}{right.text}."
                a = audit(rendered)
                checks = mechanical_admission_checks(rendered, min_letters=30, max_letters=240)
                row = {
                    "rendered": rendered,
                    "left_event": left.__dict__,
                    "right_event": right.__dict__,
                    "seam": seam_name,
                    "audit": a,
                    "mechanical_checks": checks,
                    "independent_exact": a["two_pointer_exact"],
                    "mechanically_admitted": a["two_pointer_exact"] and all(checks.values()),
                    "reader_status": "not_run; programmatic checks do not certify readability",
                }
                if a["two_pointer_exact"]:
                    exact_rows.append(row)
                if len(rows) < max_probes:
                    rows.append(row)
                if checked >= max_probes:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break
    admitted = [r for r in exact_rows if r["mechanically_admitted"]]
    new_admitted = [r for r in admitted if tape(r["rendered"]) not in KNOWN_BASELINE_TAPES]
    return {
        "status": "semantic_event_argument_csp_bounded",
        "experiment_id": "semantic-event-argument-csp-20260918",
        "signature": "semantic-event-frame|typed-argument-selection|independent-settings|pre-render-csp",
        "config": {"event_count": len(events), "seam_count": len(SEAMS), "max_probes": max_probes},
        "stats": {
            "event_frames": len(events), "pair_worlds_checked": checked,
            "bounded": checked < len(events) * len(SEAMS) * len(events),
            "stored_probes": len(rows), "exact": len(exact_rows),
            "mechanically_admitted": len(admitted),
            "new_mechanically_admitted": len(new_admitted), "reader_eligible": 0,
        },
        "rendered_candidates_and_probes": rows,
        "exact_candidates": exact_rows,
        "admitted": admitted,
        "new_admitted": new_admitted,
        "provenance": {
            "finished_tape_reversed": False,
            "catalogue_text_imported": False,
            "word_order_mirror": False,
            "independent_validator": "two-pointer normalized tape plus forward/reverse SHA-256",
            "human_readability_certified": False,
        },
        "next_repair": "carry the seam residual through typed argument tokens before complete-frame joining, then add held-out event roles",
        "reader_gate": "closed until an exact row survives intact-prose review and randomized blinded controls",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-probes", type=int, default=10_000)
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"refusing to overwrite existing output: {args.out}")
    result = run(max_probes=args.max_probes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
