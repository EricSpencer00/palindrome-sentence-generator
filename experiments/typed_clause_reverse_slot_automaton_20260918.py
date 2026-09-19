"""Reverse-slot clause automaton for readable exact palindromes.

This is a new repair of the failed typed-clause automaton.  The earlier lane
paired the same grammatical slot on both sides, which made the first character
obligation almost always impossible.  Here the right clause is expanded from
its outer edge in reverse grammatical order (ADJUNCT, OBJECT, VERB, SUBJECT),
while agreement features are carried until their governing slot is chosen.

The live state is a character residual, not a completed-tape reversal.  Every
terminal is independently pointer/hash audited and then passed through the
shared mechanical admission gate; no programmatic score certifies prose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

LEFT_SLOTS = ("DET_SUBJ", "VERB", "DET_OBJ", "ADJUNCT")
RIGHT_BUILD_SLOTS = tuple(reversed(LEFT_SLOTS))

DETS = {
    "DET_SUBJ": (("the", "SG"), ("a", "SG"), ("our", "PL"),
                 ("one", "SG"), ("this", "SG"), ("my", "SG")),
    "DET_OBJ": (("a", "SG"), ("one", "SG"), ("our", "PL"),
                ("the", "SG"), ("my", "SG")),
}
VERBS = (("carries", "SG"), ("draws", "SG"), ("helps", "SG"),
         ("marks", "SG"), ("reads", "SG"), ("sends", "SG"),
         ("carry", "PL"), ("draw", "PL"))
SUBJECTS = {
    "SG": ("baker", "doctor", "farmer", "guard", "teacher", "writer", "pilot"),
    "PL": ("bakers", "doctors", "farmers", "guards", "teachers", "writers", "pilots"),
}
OBJECTS = {
    "SG": ("letter", "message", "map", "memo", "parcel", "story", "signal"),
    "PL": ("letters", "messages", "maps", "memos", "parcels", "stories", "signals"),
}
# The first right-side slot is met from the outside edge, so its final letter
# must be able to meet a subject determiner on the left.  These are complete
# adjuncts chosen for that boundary (a/t/o/m), rather than arbitrary filler.
ADJUNCTS = (
    "home", "today", "outside", "at noon", "by dawn", "in town",
    "in a villa", "to echo", "at night", "from them", "by sea",
)


def tape(text: str) -> str:
    return normalize_letters(text)


def audit(text: str) -> dict:
    t = tape(text)
    rev = t[::-1]
    return {
        "letters": len(t),
        "two_pointer_exact": t == rev,
        "mismatch_count": sum(a != b for a, b in zip(t, rev)) + abs(len(t) - len(rev)),
        "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest(),
        "sha_equal_under_reversal": hashlib.sha256(t.encode()).hexdigest()
        == hashlib.sha256(rev.encode()).hexdigest(),
    }


def _choices(slot: str, features: dict):
    if slot in DETS:
        for det, number in DETS[slot]:
            nouns = SUBJECTS[number] if slot == "DET_SUBJ" else OBJECTS[number]
            for noun in nouns:
                if slot == "DET_SUBJ" and features.get("verb_num") not in (None, number):
                    continue
                yield f"{det} {noun}", {"subj_num": number} if slot == "DET_SUBJ" else {"obj_num": number}
    elif slot == "VERB":
        for verb, number in VERBS:
            if features.get("subj_num") not in (None, number):
                continue
            yield verb, {"verb_num": number}
    elif slot == "ADJUNCT":
        for phrase in ADJUNCTS:
            yield phrase, {}
    else:
        raise ValueError(slot)


def _consume(side: str, residual: str, left_piece: str, right_piece: str):
    """Compare the two outside-in character streams and retain unmatched debt."""
    left_stream = tape(left_piece)
    right_stream = tape(right_piece)[::-1]
    if side == "L":
        left_stream = residual + left_stream
    elif side == "R":
        right_stream = residual + right_stream
    k = min(len(left_stream), len(right_stream))
    if left_stream[:k] != right_stream[:k]:
        return None
    if len(left_stream) > len(right_stream):
        return "L", left_stream[k:]
    if len(right_stream) > len(left_stream):
        return "R", right_stream[k:]
    return "", ""


def run(*, max_states: int = 2_000_000) -> dict:
    # state: side, residual, left_words, right_build_words, feature signatures
    states = {("", "", (), (), (), ()): ({}, {})}
    counts = [len(states)]
    pruned = Counter()
    for left_slot, right_slot in zip(LEFT_SLOTS, RIGHT_BUILD_SLOTS):
        nxt = {}
        for (side, residual, left_words, right_words, left_sig, right_sig), (lf, rf) in states.items():
            for lp, lfeat in _choices(left_slot, lf):
                for rp, rfeat in _choices(right_slot, rf):
                    got = _consume(side, residual, lp, rp)
                    if got is None:
                        pruned["character_obligation_conflict"] += 1
                        continue
                    ns, nd = got
                    nlf = {**lf, **lfeat}
                    nrf = {**rf, **rfeat}
                    # Keep one lexical witness for each residual/type state;
                    # this is a finite product, not a duplicate seed sweep.
                    key = (
                        ns, nd, left_words + (lp,), right_words + (rp,),
                        tuple(sorted(nlf.items())), tuple(sorted(nrf.items())),
                    )
                    nxt.setdefault(key, (nlf, nrf))
                    if len(nxt) >= max_states:
                        pruned["state_budget"] += 1
                        break
                if len(nxt) >= max_states:
                    break
            if len(nxt) >= max_states:
                break
        states = nxt
        counts.append(len(states))
        if not states:
            break

    rows = []
    for (side, residual, left_words, right_words, _ls, _rs), (_lf, _rf) in states.items():
        if side or residual:
            continue
        rendered = " ".join(left_words + tuple(reversed(right_words))) + "."
        t = tape(rendered)
        checks = mechanical_admission_checks(rendered, min_letters=30, max_letters=220)
        rows.append({
            "rendered": rendered,
            "left_clause": list(left_words),
            "right_clause": list(reversed(right_words)),
            "letters": len(t),
            "normalized_tape": t,
            "independent_exact": bool(t) and t == t[::-1],
            "audit": audit(rendered),
            "mechanical_checks": checks,
            "mechanically_admitted": bool(t) and t == t[::-1] and all(checks.values()),
            "reader_status": "not_run; programmatic checks do not certify readability",
        })
    rows.sort(key=lambda r: (-r["mechanically_admitted"], -r["letters"], r["rendered"]))
    admitted = [r for r in rows if r["mechanically_admitted"]]
    return {
        "status": "reverse_slot_typed_clause_complete",
        "experiment_id": "typed-clause-reverse-slot-automaton-20260918",
        "signature": "typed-clause-reverse-slot-order|agreement-carry|live-character-residual|independent-audit",
        "config": {"left_slots": LEFT_SLOTS, "right_build_slots": RIGHT_BUILD_SLOTS, "max_states": max_states},
        "stats": {"state_counts": counts, "terminal_exact": len(rows), "mechanically_admitted": len(admitted),
                   "reader_eligible": 0, "pruned": dict(pruned)},
        "rendered_candidates_and_probes": rows,
        "admitted": admitted,
        "provenance": {
            "source": "hand-authored typed clause inventory; no catalogue text or seed scaffold",
            "finished_tape_reversed": False,
            "independent_validator": "two-pointer normalized tape plus forward/reverse SHA-256",
            "human_readability_certified": False,
        },
        "next_repair": "replace lexical subjects/objects with hand-authored event frames while preserving reverse grammatical slot order and live residual",
        "reader_gate": "closed until intact English prose survives manual review and randomized blinded controls",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-states", type=int, default=2_000_000)
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"refusing to overwrite existing output: {args.out}")
    result = run(max_states=args.max_states)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
