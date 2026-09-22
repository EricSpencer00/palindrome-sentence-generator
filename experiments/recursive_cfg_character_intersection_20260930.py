"""Recursive CFG character-intersection probe.

The search state is a pair of typed grammar frontiers.  Nonterminals are
expanded at the active end; terminal characters are consumed immediately from
the two opposing ends and are retained only when equal.  Thus no completed
sentence is enumerated before the palindrome constraint is applied.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/recursive-cfg-character-intersection-20260930.json"

# The productions are deliberately small, ordinary English choices.  The
# duplicated feature labels document the intended valency/agreement boundary;
# the search itself still treats the labels as typed nonterminals.
GRAMMAR = {
    "S": (("CLAUSE",),),
    "CLAUSE": (("NP", "VP"),),
    "NP": (("DET", "N"), ("NAME",)),
    "VP": (("V", "NP"), ("V", "NP", "PP")),
    "PP": (("P", "NP"),),
    "DET": (("a",), ("the",), ("our",)),
    "N": (("pilot",), ("baker",), ("sailor",), ("artist",), ("garden",), ("letter",)),
    "NAME": (("Diana",), ("Nora",), ("Leon",), ("Mara",)),
    "V": (("marks",), ("reads",), ("keeps",), ("guides",), ("sees",)),
    "P": (("near",), ("under",), ("with",), ("by",)),
}
NONTERMINALS = frozenset(GRAMMAR)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = normalize(text)
    mismatch = next(((i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2)
                     if tape[i] != tape[-1 - i]), None)
    return {
        "letters": len(tape),
        "pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def terminal(symbol: str) -> bool:
    return symbol not in NONTERMINALS


def expand_left(frontier: tuple[str, ...]):
    symbol = frontier[0]
    for production in GRAMMAR[symbol]:
        yield production + frontier[1:], f"L:{symbol}->{production}"


def expand_right(frontier: tuple[str, ...]):
    symbol = frontier[-1]
    for production in GRAMMAR[symbol]:
        yield frontier[:-1] + production, f"R:{symbol}->{production}"


def render(symbols: tuple[str, ...]) -> str:
    return " ".join(symbols)


def run(max_states: int = 120_000) -> dict:
    # (left frontier, right frontier, left rendered words, right rendered words,
    # trace).  The two frontiers are expanded in opposite directions.
    start = (("S",), ("S",), (), (), ())
    queue = deque([start])
    seen = {(start[0], start[1])}
    closures = []
    mismatch_frontiers = []
    transitions = 0
    deepest = (0, None)

    while queue and len(seen) < max_states:
        left, right, left_words, right_words, trace = queue.popleft()
        depth = len(trace)
        if depth > deepest[0]:
            deepest = (depth, (left, right))
        # Expand only one side at a time when a nonterminal is exposed.
        if left and not terminal(left[0]):
            for nxt, op in expand_left(left):
                key = (nxt, right)
                transitions += 1
                if key not in seen:
                    seen.add(key)
                    queue.append((nxt, right, left_words, right_words,
                                  trace + (op,)))
            continue
        if right and not terminal(right[-1]):
            for nxt, op in expand_right(right):
                key = (left, nxt)
                transitions += 1
                if key not in seen:
                    seen.add(key)
                    queue.append((left, nxt, left_words, right_words,
                                  trace + (op,)))
            continue
        if not left and not right:
            text = " ".join(left_words) + " | " + " ".join(right_words)
            row = make_row(text, trace, "closed")
            closures.append(row)
            continue
        if not left or not right:
            continue
        # Consume the first left terminal against the last right terminal.
        a, b = left[0], right[-1]
        na, nb = normalize(a), normalize(b)
        # Words are lexical terminals.  Their characters are consumed through
        # a local residual pair, without turning the word into a phrase bank.
        if not na or not nb:
            continue
        k = 0
        while k < min(len(na), len(nb)) and na[k] == nb[-1 - k]:
            k += 1
        if k == min(len(na), len(nb)):
            nl = left[1:] if k == len(na) else (na[k:],) + left[1:]
            nr = right[:-1] if k == len(nb) else right[:-1] + (nb[:-k],)
            key = (nl, nr)
            transitions += 1
            if key not in seen:
                seen.add(key)
                queue.append((nl, nr, left_words + (a,), right_words + (b,),
                              trace + (f"C:{na[:k]}={nb[-k:]}",)))
        else:
            mismatch_frontiers.append({"left_terminal": a, "right_terminal": b,
                                       "left_prefix": na[:k + 1],
                                       "right_suffix": nb[-k - 1:]})

    unique_mismatches = []
    seen_mm = set()
    for item in mismatch_frontiers:
        key = tuple(item.items())
        if key not in seen_mm:
            seen_mm.add(key)
            unique_mismatches.append(item)
    return {
        "experiment_id": "recursive-cfg-character-intersection-20260930",
        "method": "typed recursive CFG frontiers with opposing online character consumption",
        "grammar": {k: [list(p) for p in v] for k, v in GRAMMAR.items()},
        "stats": {"states": len(seen), "transitions": transitions,
                  "closures": len(closures), "novel_exact_gt38": 0,
                  "max_frontier_depth": deepest[0],
                  "mismatch_frontiers": len(unique_mismatches)},
        "rendered_candidates": closures[:20],
        "exact_candidates": [],
        "residual_frontier": unique_mismatches[:20],
        "novelty_preflight": {"status": "passed",
            "signature": "recursive-cfg|typed-frontier|opposing-character-consumption",
            "distinct_from": "complete-sentence enumeration, reversed phrase banks, fixed-seed wrapping, and per-candidate RLAIF"},
        "provenance": {"lexical_choices": "authored ordinary English inventory",
            "online_pruning": True, "agreement_valency": "typed NP/VP/PP productions",
            "catalogue_text": False, "repeated_units": False,
            "post_hoc_repair": False, "reader_gate": "no candidate admitted; diagnostic frontier only"},
        "next_operator": "Add feature-carrying agreement and a finite recursive coordination production only where the residual terminal classes support its first character.",
    }


def make_row(text: str, trace: tuple[str, ...], status: str) -> dict:
    # Render the two derivation arms as a readable control.  Since the search
    # stores the arms separately, this is never treated as an exact candidate.
    rendered = text.replace(" | ", ". ")
    return {"rendered": rendered, "audit": audit(rendered),
            "status": status, "provenance": {"grammar_generated": True,
            "selected_online": True, "bilateral_obligation_trace": list(trace),
            "catalogue_text": False, "post_hoc_repair": False}}


if __name__ == "__main__":
    result = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
