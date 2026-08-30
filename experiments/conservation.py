"""How far does one placement advance the mirror?

`experiments/RESULTS-extend.md` states that a one-sided placement advances the
palindrome by 1.09 letters whatever the size of the thing placed, and cites
`RESULTS-chunking` for it. That file is not in this repository, so the figure
had no artifact behind it and is re-measured here from the search itself.

Definition, so the number means something specific. A placement takes a state
with overhang `o` and puts a unit with letters `w` on the owing side. The
letters that are settled by the move -- matched on both sides and never
revisited -- are `min(|o|, |w|)`. Everything past that becomes the new
overhang, owed by whichever side is now longer. Two quantities are reported because the original claim does not say which it
means: `settled`, the letters fixed on both sides by the move, and `net`, how
much the overhang actually shrank -- negative when a long unit leaves more debt
than it discharges.

The claim under test is that advance does not grow with `|w|`. If it does not,
placing longer units buys nothing per move and one-sided chunking is a
treadmill; if it does, the paper's argument for two-sided operations needs a
different basis.
"""
from __future__ import annotations

import statistics as st
import sys

sys.path.insert(0, ".")
from llm_palindrome.centerout import consume_suffix
from llm_palindrome.generate import build_vocab
from llm_palindrome.search import WordTries, consume, unit_letters


def measure(limit=400, vocab_size=6000, max_overhang=24, budget=60000):
    tries = WordTries(build_vocab(vocab_size))
    rows = []
    # Walk the reachable overhangs breadth-first from empty. Every legal
    # placement out of every state visited is one observation, so the sample is
    # of the search's actual move distribution rather than of a word list.
    seen, frontier = {""}, [""]
    while frontier and len(rows) < budget:
        o = frontier.pop()
        for side, cands, take in (
                ("L", tries.left_candidates(o, limit), consume),
                ("R", tries.right_candidates(o[::-1], limit), consume_suffix)):
            for w in cands:
                letters = unit_letters(w)
                res = take(letters, o)
                if res is None:
                    continue
                new_over, _ = res
                # Two readings of "advance", because the claim under test does
                # not say which it means.
                #   settled   letters fixed on both sides by this move
                #   net       how much the overhang actually shrank; negative
                #             when the unit overshoots and leaves more debt
                #             than it paid
                settled = min(len(o), len(letters))
                net = len(o) - len(new_over)
                rows.append((len(letters), settled, net))
                if len(rows) >= budget:
                    break
                if new_over not in seen and len(new_over) <= max_overhang:
                    seen.add(new_over)
                    frontier.append(new_over)
            if len(rows) >= budget:
                break
    return rows


def summarise(rows):
    by = {}
    for L, s_, n_ in rows:
        by.setdefault(min(L, 10), []).append((s_, n_))
    return [{"unit": ("10+" if L == 10 else str(L)), "n": len(by[L]),
             "settled": st.mean([x for x, _ in by[L]]),
             "net": st.mean([y for _, y in by[L]])} for L in sorted(by)]


if __name__ == "__main__":
    rows = measure()
    import json
    json.dump({"placements": len(rows),
               "mean_settled": st.mean([s_ for _, s_, _ in rows]),
               "mean_net": st.mean([n_ for _, _, n_ in rows]),
               "by_unit": summarise(rows)},
              open("experiments/conservation.json", "w"), indent=1)
    se = [s_ for _, s_, _ in rows]
    ne = [n_ for _, _, n_ in rows]
    print(f"{len(rows):,} placements")
    print(f"  mean settled {st.mean(se):.2f}  sd {st.stdev(se):.2f}")
    print(f"  mean net     {st.mean(ne):.2f}  sd {st.stdev(ne):.2f}")
    print(f"\n  {'unit letters':>12} {'n':>7} {'settled':>9} {'net':>9}")
    by = {}
    for L, s_, n_ in rows:
        by.setdefault(min(L, 10), []).append((s_, n_))
    for L in sorted(by):
        v = by[L]
        tag = f"{L}" if L < 10 else "10+"
        print(f"  {tag:>12} {len(v):>7} "
              f"{st.mean([x for x, _ in v]):>9.2f} "
              f"{st.mean([y for _, y in v]):>9.2f}")
