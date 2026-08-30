"""Punctuation as a marker applied afterwards, by inference, not during search.

`experiments/presenter_cost.py` established that the shipped presenter is worse
than plain spacing on both judges, because it chooses its cuts with a dynamic
programme whose objective pays per run and so buys fragments. The fix tried
there -- weight the gain by run length -- helps and still does not beat bare
spacing.

This arm changes what does the punctuating rather than how it scores cuts. The
search returns a bare word run; a language model is then asked to add marks and
nothing else. The model has no constraint to satisfy, because the palindrome is
already closed: it only has to decide where the sentences are.

The letters are asserted identical afterwards. A model that adds, drops or
reorders a letter has broken the palindrome, and that reply is discarded and
counted rather than repaired -- the discard rate is part of the result, since a
punctuater that only works four times in five is a different proposition from
one that always works.

Two punctuaters (20b, 120b) crossed with two judges (20b, 120b), so a model
scoring its own output above the other's shows up as an asymmetry rather than
hiding inside a single number.
"""
from __future__ import annotations

import json, random, sys

sys.path.insert(0, ".")
from llm_palindrome.present import present
from llm_palindrome.syntax import brown_tables
from llm_palindrome.validator import normalize
from experiments.coherence_scale import ask, chat
from experiments.presenter_cost import HAND, present_perword

import re, urllib.request

HOST = "http://localhost:11434/api/chat"

MARK = """Add punctuation and capitalisation to this text so it reads as
English sentences.

Rules, all of them absolute:
- Do NOT add, remove, change or reorder any letter or any word.
- Only insert or change spaces, capitals, and the marks . , : ; ! ? ' " -
- The words are already in their final order. Your job is only to decide where
  the sentences and phrases break.
- The text may be strange or archaic. Punctuate it as it stands; do not fix it.

Text:
{text}

Reply with the punctuated text and nothing else."""


def punctuate(model, words):
    out = chat(model, MARK.format(text=words)).strip()
    # Models like to wrap the answer in quotes or a code fence.
    out = re.sub(r"^```[a-z]*\n?|```$", "", out).strip().strip('"')
    if normalize(out) != normalize(words):
        return None, out
    return out, out


def build(punct_models):
    t_, s_, g_ = brown_tables()
    items, broken = [], {m: 0 for m in punct_models}
    for words, hand in HAND.items():
        assert normalize(hand) == normalize(words)
        items.append(("hand", hand))
        items.append(("bare", words))
        items.append(("present", present(words.split(), t_, s_, g_)))
        items.append(("perword", present_perword(words.split(), t_, s_, g_)))
        for m, tag in punct_models.items():
            got, raw = punctuate(m, words)
            if got is None:
                broken[m] += 1
                print(f"  BROKE {tag}: {words!r} -> {raw[:90]!r}", flush=True)
                continue
            items.append((f"llm_{tag}", got))
    random.Random(7).shuffle(items)
    return items, broken


if __name__ == "__main__":
    judge, tag = sys.argv[1], sys.argv[2]
    cache = "runs/punct/punct_after_items.json"
    try:
        blob = json.load(open(cache))
        items, broken = [tuple(x) for x in blob["items"]], blob["broken"]
        print(f"reusing {len(items)} items from {cache}")
    except FileNotFoundError:
        # Built once and cached so both judges score identical text.
        items, broken = build({"gpt-oss:20b": "20b", "gpt-oss:120b-cloud": "120b"})
        json.dump({"items": items, "broken": broken}, open(cache, "w"), indent=1)
    print("letters changed, discarded:", broken)
    rows = []
    for n, (kind, text) in enumerate(items):
        score, _ = ask(judge, text)
        rows.append({"kind": kind, "score": score, "text": text})
        print(f"{n:4d} {kind:<10} {score}  {text[:66]!r}", flush=True)
    json.dump(rows, open(f"runs/punct/after_{tag}.json", "w"), indent=1)
    print("wrote", f"runs/punct/after_{tag}.json")
