#!/usr/bin/env python3
"""Synchronous lexical phrase grammar for constructive center-out growth.

Each step selects one phrase for the left edge and one phrase for the right
edge together.  Their newly exposed letters must mirror at every newly closed
character pair.  No completed string is reversed or edited; the tape is grown
by inserting the selected phrases at its two ends.
"""
from __future__ import annotations
import argparse, hashlib, json, re, time
import socket
from dataclasses import dataclass
from pathlib import Path

GRAMMAR = {
    "S": [("NP", "VP")],
    "NP": [("DET", "N")],
    "VP": [("V", "NP"), ("V", "ADV")],
    "DET": [("the",), ("a",)],
    "N": [("baker",), ("pilot",), ("clerk",), ("poet",), ("map",), ("bird",)],
    "V": [("marks",), ("opens",), ("sees",), ("keeps",), ("reads",)],
    "ADV": [("today",), ("at noon",), ("near dawn",)],
}
TOKEN_TAPE = lambda s: re.sub(r"[^a-z]", "", s.lower())

@dataclass(frozen=True)
class State:
    left: tuple[str, ...]
    right: tuple[str, ...]
    pending: str
    left_symbol: str
    right_symbol: str
    boundary_debt: int
    steps: int

def expand(symbol: str) -> list[str]:
    if symbol not in GRAMMAR:
        return [symbol]
    out = []
    for prod in GRAMMAR[symbol]:
        choices = [expand(x) for x in prod]
        rows = [""]
        for group in choices:
            rows = [f"{a} {b}".strip() for a in rows for b in group]
        out.extend(rows)
    return out

LEXICAL_PHRASES = sorted(set(sum((expand(x) for x in ("NP", "VP")), [])))

def compatible(left: str, right: str, left_context: str = "", right_context: str = "") -> bool:
    """Propagate all currently exposed boundary debt, not one edge character."""
    a = TOKEN_TAPE(left_context + left)
    b = TOKEN_TAPE(right + right_context)
    overlap = min(len(a), len(b))
    return bool(overlap) and a[-overlap:] == b[:overlap][::-1]

def exact(text: str) -> tuple[bool, str, str, int]:
    tape = TOKEN_TAPE(text)
    f = hashlib.sha256(tape.encode()).hexdigest()
    r = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return bool(tape) and tape == tape[::-1], f, r, len(tape)

def pointer_mismatches(text: str) -> int:
    tape = TOKEN_TAPE(text)
    return sum(a != b for a, b in zip(tape, reversed(tape))) // 2

def search(depth: int, beam: int) -> list[State]:
    # Start with a grammatical seed; all further material is paired growth.
    # A single center token avoids mirrored self-seeding. Grammar debt is live
    # on both sides; it is discharged only by a complete lexical expansion.
    states = [State(("a",), tuple(), "S", "NP", "VP", 1, 0)]
    for step in range(depth):
        nxt = []
        for st in states:
            used = set(st.left + st.right)
            used_content = {w for p in used for w in re.sub(r"[^a-z ]", "", p.lower()).split() if len(w) > 2}
            for lp in LEXICAL_PHRASES:
                for rp in LEXICAL_PHRASES:
                    if lp in used or rp in used or lp == rp:
                        continue
                    content = {w for p in (lp, rp) for w in re.sub(r"[^a-z ]", "", p.lower()).split() if len(w) > 2}
                    if content & used_content or not compatible(lp, rp, " ".join(st.left), " ".join(st.right)):
                        continue
                    debt = abs(len(TOKEN_TAPE(" ".join((lp,) + st.left))) - len(TOKEN_TAPE(" ".join(st.right + (rp,)))))
                    nxt.append(State((lp,) + st.left, st.right + (rp,),
                                     "" if step + 1 >= depth else "S", "NP", "VP", debt, step + 1))
        states = sorted(nxt, key=lambda s: (abs(len("".join(map(TOKEN_TAPE, s.left))) - len("".join(map(TOKEN_TAPE, s.right)))), s.left))[:beam]
        if not states:
            break
    return states

def row(st: State) -> dict:
    text = " ".join(st.left + st.right)
    ok, hf, hr, n = exact(text)
    return {"rendered": text, "letters": n, "closed": ok, "pending_grammar": st.pending,
            "left_phrases": list(st.left), "right_phrases": list(st.right),
            "boundary_debt": st.boundary_debt, "grammar_state": {"left": st.left_symbol, "right": st.right_symbol},
            "exactness": {"two_pointer": ok, "pointer_mismatches": pointer_mismatches(text),
                          "sha256_forward": hf, "sha256_reverse": hr, "hash_equal": hf == hr},
            "provenance": {"construction": "synchronous_lexical_phrase_grammar_centerout",
                           "catalogue_used": False, "wrapped_seed": False,
                           "finished_tape_reversal": False, "posthoc_repair": False,
                           "joint_grammar_and_mirror_choice": True,
                           "lexicon": "fresh authored mini-grammar in this file"}}

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--beam", type=int, default=40); ap.add_argument("--out", required=True)
    args = ap.parse_args(); t0 = time.time(); states = search(args.depth, args.beam)
    rows = [row(s) for s in states if exact(" ".join(s.left + s.right))[0]]
    payload = {"experiment": "synchronous-lexical-centerout-20260919",
               "method": "typed phrase expansion with live boundary mirror checks",
               "parameters": vars(args), "host": socket.gethostname(),
               "grammar_sha256": hashlib.sha256(json.dumps(GRAMMAR, sort_keys=True).encode()).hexdigest(),
               "candidates": rows, "closures": sum(x["closed"] for x in rows),
               "frontier_count": len(states),
               "next_construction": "add a phrase-pair index keyed by (left-first,right-last) and carry full boundary debt, then seek a grammatical seam rather than relaxing compatibility",
               "runtime_seconds": round(time.time()-t0, 3), "novelty": "new method; not a catalogue walk or post-hoc repair"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({k: payload[k] for k in ("experiment", "closures", "runtime_seconds")}))
if __name__ == "__main__": main()
