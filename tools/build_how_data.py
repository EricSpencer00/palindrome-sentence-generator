"""Summaries from tools/paper_metrics.py output -> web/public/how/data.json.

usage: python tools/build_how_data.py runs/paper-metrics-*.json
Classic palindromes are a few words long, and GPT-2 perplexity on a text that
short measures the missing context rather than the text, so it is left out.
"""
import json
import sys

report = json.load(open(sys.argv[1]))
out = {name: dict(c["summary"]) for name, c in report.items()}
out["classics"]["gpt2_ppl_median"] = None
json.dump(out, open("web/public/how/data.json", "w"), indent=1)
