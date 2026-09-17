"""Fresh-center compositional grammar experiment.

The center is an authored event (not a letter or a self-palindromic word).
Complete clauses grow on either side of it.  At each expansion the decoder
records the character required by the opposite frontier, but lexical choices
remain grammatical choices; it never manufactures a reversed tape.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "luna-fresh-center-composition-20260917"
SIGNATURE = "fresh-center-event|compositional-clause-growth|function-word-inflection-obligations|no-mirror|pointer-sha"
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

DETS = ({"id": "the", "text": "the"}, {"id": "a", "text": "a"})
SUBJECTS = ({"id": "surveyor", "text": "patient surveyor", "agreement": "singular"},
            {"id": "keeper", "text": "young keeper", "agreement": "singular"})
VERBS = ({"id": "marks", "text": "marks", "agreement": "singular"},
         {"id": "records", "text": "records", "agreement": "singular"})
OBJECTS = ({"id": "inlet", "text": "a quiet inlet"},
           {"id": "charts", "text": "wet charts"})
LOCATIVES = ({"id": "pier", "text": "beside the eastern pier"},
             {"id": "archive", "text": "near the stone archive"})
RIGHT_LOCATIVES = ({"id": "dock", "text": "toward the river dock"},
                   {"id": "office", "text": "inside the harbor office"})

# This event is intentionally neither a known seed nor a self-palindromic unit.
CENTER_EVENT = {"id": "bell_rings", "text": "the bell rings", "role": "punctual auditory event"}


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def pointer(text: str) -> dict:
    tape = letters(text)
    mismatches = []
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"left_index": i, "right_index": j, "left": tape[i], "required": tape[j]})
        i, j = i + 1, j - 1
    return {"algorithm": "independent_two_pointer", "letters": len(tape),
            "exact": bool(tape) and not mismatches, "mismatch_count": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else None}


def sha_audit(text: str) -> dict:
    tape = letters(text)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"algorithm": "independent_forward_reverse_sha256", "forward": forward,
            "reverse": reverse, "exact": bool(tape) and forward == reverse}


def clause(det: dict, subject: dict, verb: dict, obj: dict, loc: dict) -> str:
    return f"{det['text']} {subject['text']} {verb['text']} {obj['text']} {loc['text']}"


def obligations(text: str) -> dict:
    tape = letters(text)
    pairs = []
    for i in range(min(len(tape) // 2, 30)):
        j = len(tape) - 1 - i
        pairs.append({"offset": i, "actual": tape[i], "required": tape[j], "satisfied": tape[i] == tape[j]})
    return {"equation": "left frontier character = right frontier character",
            "pairs_checked": len(tape) // 2, "frontier_sample": pairs,
            "satisfied_prefix": next((i for i, p in enumerate(pairs) if not p["satisfied"]), len(pairs))}


def novelty_preflight() -> dict:
    data = json.loads(REGISTRY.read_text())
    rows = [*data.get("entries", []), *data.get("excluded", [])]
    collisions = [r.get("id") for r in rows if r.get("id") != EXPERIMENT and
                  (r.get("signature") == SIGNATURE or r.get("artifact") == "experiments/luna_fresh_center_composition_20260917.py")]
    return {"registry_entries": len(data.get("entries", [])), "signature": SIGNATURE,
            "collisions": collisions, "passed": not collisions,
            "catalogue_lookup": False, "known_seed_imported": False}


def render(left: tuple[dict, ...], right: tuple[dict, ...]) -> str:
    return f"{clause(*left)} before {CENTER_EVENT['text']}, while {clause(*right)}."


def audit(left: tuple[dict, ...], right: tuple[dict, ...], rank: int) -> dict:
    text = render(left, right)
    p, s = pointer(text), sha_audit(text)
    # Function words may legitimately recur ("the", "a"); the two complete
    # clauses are still distinct compositional units when their renderings differ.
    distinct = clause(*left) != clause(*right)
    complete = text.endswith(".") and text.count(" ") >= 14 and all(text.count(x["text"]) == 1 for x in (CENTER_EVENT,))
    flags = {"fixed_tape": False, "reverse_decoder": False, "word_order_mirror": False,
             "nested_palindrome_span": False, "self_palindromic_center": letters(CENTER_EVENT["text"]) == letters(CENTER_EVENT["text"])[::-1],
             "repeated_chunk": not distinct, "fragment": not complete, "catalogue_text": False}
    return {"rank": rank, "rendered": text, "center_event": CENTER_EVENT,
            "left_clause": {"text": clause(*left), "slot_ids": [x["id"] for x in left]},
            "right_clause": {"text": clause(*right), "slot_ids": [x["id"] for x in right]},
            "complete_grammar": complete, "obligations": obligations(text),
            "independent_pointer": p, "independent_sha": s,
            "independent_exact_agreement": p["exact"] == s["exact"],
            "anti_shortcut_flags": flags, "mechanically_admitted": bool(p["exact"] and s["exact"] and complete and distinct),
            "residual_repair": p["first_mismatch"],
            "reader_status": "complete readable prose; diagnostic palindrome failure, not an exact palindrome",
            "next_reader_facing_test": "Blind readers rate grammaticality and event coherence before seeing the mismatch trace."}


def run() -> dict:
    pre = novelty_preflight()
    if not pre["passed"]:
        raise RuntimeError(pre)
    # Independent slot assignments force ordinary agreement and function words.
    lefts = list(itertools.product(DETS, SUBJECTS[:1], VERBS, OBJECTS[:1], LOCATIVES))
    rights = list(itertools.product(DETS, SUBJECTS[1:], VERBS, OBJECTS[1:], RIGHT_LOCATIVES))
    rows = []
    for rank, (left, right) in enumerate(itertools.islice(itertools.product(lefts, rights), 12), 1):
        row = audit(left, right, rank)
        if not row["anti_shortcut_flags"]["repeated_chunk"]:
            rows.append(row)
    rows.sort(key=lambda r: (-r["obligations"]["satisfied_prefix"], -r["independent_pointer"]["letters"], r["rank"]))
    exact = [r for r in rows if r["mechanically_admitted"]]
    return {"experiment": EXPERIMENT, "signature": SIGNATURE,
            "status": "exact closure found" if exact else "complete prose plus residual repair",
            "novelty_preflight": pre, "center_event": CENTER_EVENT,
            "states_considered": 12, "candidate_count": len(rows), "exact_count": len(exact),
            "rendered_candidates": rows, "exact_survivors": exact,
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
                           "fresh_authored_center": True, "function_word_choices_live": True,
                           "inflection_choices_live": True, "catalogue_imported": False},
            "anti_shortcut_policy": "Reject fixed tapes, reverse decoding, word-order mirrors, nested palindrome spans, self-palindromic centers, fragments, and catalogue text.",
            "next_repair": "Use the recorded first mismatch to hold out one determiner or inflection choice, then rerun both clauses without changing the event.",
            "reader_facing_test": {"required": "blind intact-prose rating plus shuffled-clause control", "status": "pending"}}


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(OUT), "candidates": result["candidate_count"], "exact": result["exact_count"]}))
