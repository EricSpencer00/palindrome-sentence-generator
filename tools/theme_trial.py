"""Run the production v5 search once per theme and write one JSON line each.

usage: python tools/theme_trial.py OUT.jsonl WORKERS THEME [THEME ...]
A theme is comma-separated words; "-" means no theme (an empty prompt).
"""
import json
import multiprocessing as mp
import sys
import time


def init():
    from server import v5
    v5.warm()


def one(theme: str) -> dict:
    from server import v5
    t0 = time.time()
    prompt = "" if theme == "-" else theme.replace(",", " ")
    words = v5.search(prompt, [])
    return {"theme": theme, "text": " ".join(words), "secs": round(time.time() - t0, 1)}


if __name__ == "__main__":
    out, workers, themes = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    with mp.get_context("spawn").Pool(workers, initializer=init) as pool, open(out, "w") as fh:
        for r in pool.imap_unordered(one, themes):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(r["theme"], r["secs"], r["text"][:90], flush=True)
