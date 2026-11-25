"""Independent fluency check: GPT-2 (124M) perplexity per text.

The search never sees GPT-2, so unlike the trigram model and the spectrum
verifier it cannot have been optimised against.

usage: python tools/paper_gpt2.py metrics.json   (adds "gpt2_ppl" to every row and summary)
"""
import json
import math
import statistics as st
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

path = sys.argv[1]
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
torch.set_num_threads(2)


@torch.no_grad()
def ppl(text: str) -> float:
    ids = tok(text, return_tensors="pt").input_ids[:, :1024]
    return math.exp(model(ids, labels=ids).loss.item())


report = json.load(open(path))
for name, corpus in report.items():
    for r in corpus["rows"]:
        r["gpt2_ppl"] = ppl(r["text"])
    vals = [r["gpt2_ppl"] for r in corpus["rows"]]
    corpus["summary"]["gpt2_ppl_median"] = round(st.median(vals), 1)
    print(name, corpus["summary"]["gpt2_ppl_median"], flush=True)
json.dump(report, open(path, "w"))
