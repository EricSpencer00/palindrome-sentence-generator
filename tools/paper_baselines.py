"""Baseline corpora for tools/paper_metrics.py.

usage: python tools/paper_baselines.py OUTDIR
Writes brown.jsonl (50 contiguous ~100-word passages of the Brown corpus,
fixed seed) and classics.jsonl (the human palindromes the spectrum anchors on).
"""
import json
import random
import sys
from pathlib import Path

from nltk.corpus import brown

from llm_palindrome.spectrum import CLASSICS

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = random.Random(0)
sents = brown.sents()
with open(out / "brown.jsonl", "w") as f:
    for _ in range(50):
        i, words = rng.randrange(len(sents) - 40), []
        while len(words) < 100:
            words += sents[i]
            i += 1
        text = " ".join(words).replace(" ,", ",").replace(" .", ".").replace(" ;", ";")
        f.write(json.dumps({"text": text}) + "\n")
with open(out / "classics.jsonl", "w") as f:
    for c in CLASSICS:
        f.write(json.dumps({"text": c}) + "\n")
