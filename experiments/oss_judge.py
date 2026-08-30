"""Can gpt-oss stand in for blind judging?

Four automated proxies in this project have disagreed with blind human ranking
and none has ever agreed (docs/NORTH-STAR.md). Those were all *scorers* —
perplexity, cohesion — which read a number off a model. A chat model asked to
choose is a different instrument, so the pattern is worth re-testing rather
than assumed.

Two datasets where blind judging already returned a unanimous answer are used
as ground truth, so the model's job is to reproduce a verdict, not to set one:

  seams  27 items, single chunk vs a nest of k=2/4/8 (RESULTS-seams.md).
         Blind: single preferred 27/27, calibration 6/6.
  grow   26 items, seed vs greedily extended (RESULTS-extend.md).
         Blind: seed preferred 20/20.

Controls, which are arms and not extras:

  orientation  every item is asked twice, sides swapped. A model answering by
               position agrees with itself 0/n; one reading the text agrees
               n/n. Same control the human judges were given.
  calibration  real prose against its own shuffle, inside both sets. A model
               that cannot beat a shuffle has no power and its verdicts on the
               palindromes mean nothing, however they come out.
"""
from __future__ import annotations

import json, re, sys, urllib.request

HOST = "http://localhost:11434/api/chat"

PROMPT = """You are an annotator on a text-quality rating task.

Below are two passages, A and B. Decide: which passage reads more like English?

Judge only how well the words hang together into phrases and clauses, PER WORD
READ. Many passages are degraded, some severely. Pick one side; if the pair
seems equally bad, pick whichever has even slightly more that holds together.
Never answer "neither".

Ignore length completely. The passages are often very different lengths and
longer is NOT better. A short passage that hangs together beats a long one
that does not.

A: {a}

B: {b}

Answer with exactly one character, A or B. No reasoning, no punctuation."""


def parse_items(path):
    """The blind files are what the human judges read, verbatim."""
    text = open(path).read()
    out = {}
    for block in text.split("### ")[1:]:
        head, rest = block.split("\n", 1)
        a = rest.split("\nB: ")[0].removeprefix("A: ").strip()
        b = rest.split("\nB: ")[1].strip()
        out[int(head)] = (a, b)
    return out


def ask(model, a, b):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(a=a, b=b)}],
        "stream": False,
        # think low: the default chain of thought on this task runs for
        # minutes per item and the answer is one character. keep_alive stops
        # the model being evicted and reloaded between items, which was most
        # of the wall time in the first attempt at this arm.
        "think": "low",
        "keep_alive": "30m",
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(HOST, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        msg = json.load(r)["message"]["content"]
    # The reply is a single letter when the model complies and a sentence when
    # it does not; take the last standalone A or B either way, and record a
    # miss rather than guessing, so refusals cannot be scored as agreement.
    hits = re.findall(r"\b([AB])\b", msg)
    return (hits[-1] if hits else None), msg.strip()[:120]


if __name__ == "__main__":
    model = sys.argv[1]
    tag = sys.argv[2]
    out = {}
    for setname, j1, j2 in (("seam", "/tmp/seam_j1.txt", "/tmp/seam_j2.txt"),
                            ("grow", "/tmp/grow_j1.txt", "/tmp/grow_j2.txt")):
        for orient, path in (("fwd", j1), ("rev", j2)):
            items = parse_items(path)
            picks = {}
            for i in sorted(items):
                a, b = items[i]
                pick, raw = ask(model, a, b)
                picks[i] = pick
                print(f"{setname} {orient} {i:3d} {pick}  {raw!r}", flush=True)
            out[f"{setname}.{orient}"] = picks
    json.dump(out, open(f"runs/punct/oss_{tag}.json", "w"), indent=1)
    print("wrote", f"runs/punct/oss_{tag}.json")
