"""Outer-boundary residual search over two independently typed clauses.

The search does not mirror words or phrases.  It composes a complete clause on
each side from finite POS inventories, then zips the *character* residuals
from the outside inward.  Word boundaries may be crossed at any character;
the final matched pair may therefore close inside a word.  The right clause
is parsed in its natural order after the zipper reverses its selected words.

This is an experiment, not a readability certificate: a candidate is only
reader-facing after blinded human review.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT = Path(__file__).resolve().parents[1]
MIN_LETTERS = 39


LEFT = (
    ("det", "a the one my".split()),
    ("adj", "quiet young kind eager clever honest red small bright patient".split()),
    ("subj", "artist baker captain dancer doctor farmer guard poet sailor teacher".split()),
    ("verb", "admires carries follows gathers guides notices opens paints repairs saves".split()),
    ("det", "a the one my".split()),
    ("obj", "bridge candle garden letter mirror parcel picture river signal window".split()),
)

# This is a separate inventory, with no positional reuse of the left words.
RIGHT = (
    ("det", "a the one her his our".split()),
    ("adj", "ancient blue calm distant gentle green hidden little narrow silver".split()),
    ("subj", "author child clerk sailor singer student traveler worker".split()),
    ("verb", "answers builds cleans draws finds hears keeps learns reads".split()),
    ("det", "a the one her his our".split()),
    ("obj", "anchor basket branch compass field flower harbor idea message photo portrait radio room stone tower".split()),
)


@dataclass(frozen=True)
class Node:
    words: tuple[str, ...]
    prefix: str = ""
    index: int = 0


def _trie(words: tuple[str, ...]):
    root: dict = {"$": False}
    for word in words:
        node = root
        for char in word:
            node = node.setdefault(char, {"$": False})
        node["$"] = True
    return root


def _step(trie: dict, prefix: str, char: str):
    node = trie
    for item in prefix:
        node = node.get(item)
        if node is None:
            return None
    return node.get(char)


def _advance(trie: dict, prefix: str, char: str):
    node = _step(trie, prefix, char)
    return None if node is None else prefix + char


def _audit(text: str, left: tuple[str, ...], right_inward: tuple[str, ...], trace):
    tape = normalize_letters(text)
    exact = bool(tape) and tape == tape[::-1]
    left_ok = len(left) == len(LEFT) and all(w in LEFT[i][1] for i, w in enumerate(left))
    right_rendered = tuple(reversed(right_inward))
    right_ok = len(right_rendered) == len(RIGHT) and all(w in RIGHT[i][1] for i, w in enumerate(right_rendered))
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=220)
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_sha256": sha256(tape.encode()).hexdigest(),
        "independent_exact": exact,
        "independent_clause_parse": {"left": left_ok, "right": right_ok},
        "left_words": list(left),
        "right_words": list(right_rendered),
        "right_selected_outward_to_inward": list(right_inward),
        "pair_trace": trace,
        "mechanical_admission": central,
        "mechanically_admitted": exact and left_ok and right_ok and all(central.values()),
        "reader_status": "unreviewed; programmatic checks do not certify readability",
    }


def run(max_states: int = 500_000):
    # Each side is traversed from its outermost word toward its centre.  A
    # reverse trie on the natural right clause means the outward right word is
    # RIGHT[-1], then RIGHT[-2], etc.; its rendered order is restored at end.
    left_tries = tuple(_trie(tuple(words)) for _, words in LEFT)
    right_tries = tuple(_trie(tuple(words),) for _, words in RIGHT)
    # Trie lookup on a right word consumes its characters backwards.
    right_tries = tuple(_trie(tuple(word[::-1] for word in words)) for _, words in RIGHT)
    states = [(0, len(RIGHT) - 1, "", "", (), (), "", 0)]
    seen = set()
    candidates = []
    dead = []
    max_depth = 0
    expansions = 0
    while states and expansions < max_states:
        li, ri, lp, rp, lw, rw, tape_prefix, depth = states.pop()
        key = (li, ri, lp, rp, lw, rw)
        if key in seen:
            continue
        seen.add(key); expansions += 1; max_depth = max(max_depth, depth)
        if li == len(LEFT) and ri < 0 and not lp and not rp:
            text = " ".join(lw + tuple(reversed(rw)))
            row = _audit(text, lw, rw, [])
            if row["letters"] >= MIN_LETTERS and row["independent_exact"]:
                candidates.append(row)
            continue
        if li >= len(LEFT) or ri < 0:
            dead.append({"li": li, "ri": ri, "left_prefix": lp, "right_prefix": rp, "depth": depth, "reason": "one grammar exhausted before the other"})
            continue
        ltrie, rtrie = left_tries[li], right_tries[ri]
        lnode = ltrie
        for char in lp:
            lnode = lnode.get(char, {})
        rnode = rtrie
        for char in rp:
            rnode = rnode.get(char, {})
        common = sorted(set(lnode).intersection(rnode) - {"$"})
        if not common:
            dead.append({"li": li, "ri": ri, "left_prefix": lp, "right_prefix": rp, "depth": depth, "reason": "outer character mismatch", "left_next": sorted(set(lnode)-{"$"}), "right_next": sorted(set(rnode)-{"$"})})
            continue
        for char in common:
            nlp, nrp = lp + char, rp + char
            lnode2, rnode2 = lnode[char], rnode[char]
            lterm, rterm = bool(lnode2.get("$")), bool(rnode2.get("$"))
            # A terminal is allowed to stop or continue, independently. This
            # is the word-interior centre case: no boundary synchronization is
            # assumed, and only simultaneous full closure is accepted.
            lopts = (False, True) if lterm else (False,)
            ropts = (False, True) if rterm else (False,)
            for stop_l in lopts:
                for stop_r in ropts:
                    nli, nri, nlw, nrw = li, ri, lw, rw
                    fl, fr = nlp, nrp
                    if stop_l:
                        nli += 1; nlw += (fl,); fl = ""
                    if stop_r:
                        nri -= 1; nrw += (fr[::-1],); fr = ""
                    states.append((nli, nri, fl, fr, nlw, nrw, tape_prefix + char, depth + 1))
    # Keep frontier useful rather than retaining a potentially huge ledger.
    dead_sorted = sorted(dead, key=lambda row: (row["depth"], row["li"] + (len(RIGHT)-1-row["ri"])), reverse=True)
    return {
        "status": "outer_boundary_residual_search",
        "config": {"min_letters": MIN_LETTERS, "max_states": max_states, "center_may_be_inside_word": True, "independent_left_right_inventories": True, "whole_word_mirror_generation": False, "catalogue_text": False},
        "inventory": {"left": [{"role": r, "words": w} for r, w in LEFT], "right": [{"role": r, "words": w} for r, w in RIGHT]},
        "stats": {"states": expansions, "unique_states": len(seen), "max_depth": max_depth, "dead_frontiers": len(dead), "exact_candidates": len(candidates), "search_exhausted": not states and expansions < max_states},
        "rendered_candidates": candidates,
        "mismatch_frontier": dead_sorted[:50],
        "next_construction": "Retain the deepest typed clause prefixes and add a jointly licensed synonym at the first outer-character mismatch; do not widen with arbitrary fragments or mirror pairs.",
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "source": "task-authored finite POS inventories; independently composed clause slots"},
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--max-states", type=int, default=500_000); args = ap.parse_args()
    if args.out.exists(): ap.error("refusing to overwrite existing output")
    result = run(args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
