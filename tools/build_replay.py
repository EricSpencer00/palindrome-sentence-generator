"""Rebuild web/public/replay.json: paragraphs the page plays when the API is
down, from the current engine with an empty prompt (so each draws a theme).

usage: PYTHONPATH=. python tools/build_replay.py [N] [WORKERS]
"""
import json
import multiprocessing as mp
import os
import re
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "public", "replay.json")


def init():
    from server import v5
    v5.warm()


def one(_):
    from server import app
    return app._search_v5("")


def exact(d: dict) -> bool:
    mid = d.get("centerDisplay") or d.get("center") or ""
    s = re.sub("[^a-z]", "", " ".join(d["left"] + [mid] + d["right"]).lower())
    return bool(s) and s == s[::-1]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    with mp.get_context("spawn").Pool(workers, initializer=init) as pool:
        got = [d for d in pool.imap_unordered(one, range(n)) if d and exact(d)]
    json.dump(got, open(OUT, "w"))
    print(f"{len(got)}/{n} exact paragraphs -> {OUT}")
