"""How much of our texts' unreadability is our own punctuation?

`experiments/coherence_scale.py` measures the presenter's cost on WikiText
prose: real English scores 1.10 raw and 0.45 through `present.py`, on a scale
whose entire prose-to-shuffle range is 1.00. That is a large number, but prose
is out of distribution for a presenter built to punctuate palindrome word runs,
so it may overstate the cost on the material that matters.

This arm measures it on that material directly. The same 26 catalogued
palindromes are scored three ways:

  hand      punctuated by hand, the way these are printed where they are
            published. Written out below rather than loaded, because the bank
            stores letters and word splits only. This is the ceiling.
  present   the same word sequence through `present.py` as shipped.
  perword   the same, with the segmentation objective weighted by run LENGTH
            rather than per run. The shipped objective adds a fixed gain for
            every run it creates, so splitting is free and the dynamic
            programme buys fragments: `some men interpret nine memos` scores
            4.0 whole and 4.6 as `some men interpret | nine | memos`. Weighting
            by length removes that incentive.
  bare      the words with single spaces and no marks at all, which is the
            floor: it isolates how much any punctuation is worth here.

Every variant has identical letters, so the palindrome is unaffected and the
only thing varying is the marks. If `hand` beats `present` by a wide margin,
a share of this project's readability problem is presentation and not search.
"""
from __future__ import annotations

import json, random, sys

sys.path.insert(0, ".")
from llm_palindrome.present import MAX_RUN, WEIGHT, _tier, present, spell
from llm_palindrome.syntax import brown_tables
from llm_palindrome.validator import normalize
from experiments.coherence_scale import ask

# Hand punctuation. These are set the way the palindrome is normally printed;
# where a text has no standard printing, it is punctuated the way the sentence
# plainly wants to go. No letters are added, removed or reordered — asserted
# below rather than trusted.
HAND = {
 "doc note i dissent a fast never prevents a fatness i diet on cod":
   "Doc, note: I dissent. A fast never prevents a fatness. I diet on cod.",
 "are we not drawn onward we few drawn onward to new era":
   "Are we not drawn onward, we few, drawn onward to new era?",
 "straw no too stupid a fad i put soot on warts":
   "Straw? No, too stupid a fad. I put soot on warts.",
 "satan oscillate my metallic sonatas":
   "Satan, oscillate my metallic sonatas!",
 "cigar toss it in a can it is so tragic":
   "Cigar? Toss it in a can. It is so tragic.",
 "do nine men interpret nine men i nod":
   "Do nine men interpret? Nine men, I nod.",
 "no pet so tragic as a cigar to step on":
   "No pet so tragic as a cigar to step on.",
 "sums are not set as a test on erasmus":
   "Sums are not set as a test on Erasmus.",
 "sir i demand i am a maid named iris":
   "Sir, I demand: I am a maid named Iris.",
 "stressed was i ere i saw desserts":
   "Stressed was I ere I saw desserts.",
 "golf no sir prefer prison flog":
   "Golf? No, sir. Prefer prison. Flog!",
 "norma is as selfless as i am ron":
   "Norma is as selfless as I am, Ron.",
 "some men interpret nine memos":
   "Some men interpret nine memos.",
 "a tin mug for a jar of gum nit a":
   "A tin mug for a jar of gum, nit a.",
 "campus motto bottoms up mac":
   "Campus motto: bottoms up, Mac!",
 "drab as a fool aloof as a bard":
   "Drab as a fool, aloof as a bard.",
 "no sir away a papaya war is on":
   "No, sir! Away! A papaya war is on!",
 "a dog a plan a canal pagoda":
   "A dog, a plan, a canal: Pagoda.",
 "a man a plan a canal panama":
   "A man, a plan, a canal: Panama.",
 "eva can i see bees in a cave":
   "Eva, can I see bees in a cave?",
 "go deliver a dare vile dog":
   "Go deliver a dare, vile dog!",
 "may a moody baby doom a yam":
   "May a moody baby doom a yam?",
 "murder for a jar of red rum":
   "Murder for a jar of red rum.",
 "no it can assess an action":
   "No, it can assess an action.",
 "ten animals i slam in a net":
   "Ten animals I slam in a net.",
 "rats live on no evil star":
   "Rats live on no evil star.",
}


def segment_perword(words, table, shapes, trigrams):
    """`present.segment` with the gain scaled by run length."""
    n = len(words)
    best = [float("-inf")] * (n + 1)
    back = [(-1, 0)] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(max(0, j - MAX_RUN), j):
            if best[i] == float("-inf"):
                continue
            tier = _tier(words[i:j], table, shapes, trigrams)
            gain = WEIGHT[tier] * (j - i)
            if best[i] + gain > best[j]:
                best[j] = best[i] + gain
                back[j] = (i, tier)
    runs, j = [], n
    while j > 0:
        i, tier = back[j]
        runs.append((list(words[i:j]), tier))
        j = i
    return runs[::-1]


def present_perword(words, table, shapes, trigrams):
    """`present.present`'s marking rules over the length-weighted segmentation.

    Kept as a copy rather than a flag on `present` so that measuring the change
    cannot alter what the endpoint serves while the measurement is running.
    """
    from llm_palindrome.present import SENTENCE, SHAPED
    runs = segment_perword(list(words), table, shapes, trigrams)
    out, colon_used, start = [], False, True
    pending, used = [], set()
    for idx, (run, tier) in enumerate(runs):
        last = idx == len(runs) - 1
        text = spell(run, period=False)
        if not start and run[0] != "i":
            text = text[0].lower() + text[1:]
        if last or tier == SENTENCE:
            mark = "."
        elif tier == SHAPED and not colon_used:
            mark, colon_used = ":", True
        else:
            mark = ","
        if mark == "." and not last and \
                normalize(" ".join(pending + [text])) in used:
            mark = ","
        out.append(text + mark)
        pending.append(text)
        if mark in ".!?":
            used.add(normalize(" ".join(pending)))
            pending = []
        start = mark in ".!?"
    result = " ".join(out)
    assert normalize(result) == normalize(" ".join(words))
    return result


def build():
    t_, s_, g_ = brown_tables()
    items = []
    for words, hand in HAND.items():
        assert normalize(hand) == normalize(words), f"hand edit changed {words!r}"
        items.append(("hand", hand))
        items.append(("present", present(words.split(), t_, s_, g_)))
        items.append(("perword", present_perword(words.split(), t_, s_, g_)))
        items.append(("bare", words))
    random.Random(5).shuffle(items)
    return items


if __name__ == "__main__":
    model, tag = sys.argv[1], sys.argv[2]
    rows = []
    for n, (kind, text) in enumerate(build()):
        score, _ = ask(model, text)
        rows.append({"kind": kind, "score": score, "text": text})
        print(f"{n:4d} {kind:<8} {score}  {text[:70]!r}", flush=True)
    json.dump(rows, open(f"runs/punct/pcost_{tag}.json", "w"), indent=1)
    print("wrote", f"runs/punct/pcost_{tag}.json")
