#!/usr/bin/env python3
"""Deterministic ordering/listiness diagnostic for a six-pair ABBA family."""
import itertools, json, random, re
from pathlib import Path

OUT = Path("runs/paragraph-family-observation-state-diagnostic-20261002.json")
PAIRS = [("desserts", "stressed"), ("lager", "regal"), ("war", "raw"),
         ("trams", "smart"), ("guns", "snug"), ("gums", "smug")]
DOMAIN = {"trams": 0, "guns": 1, "war": 2, "lager": 3, "desserts": 4, "gums": 5}
STATE = {"gums", "smug", "stressed", "snug", "regal", "raw"}

def score(order):
    # Reward an observation-first progression, penalize abrupt identity repetition.
    ranks = [DOMAIN[a] for a, _ in order]
    monotonic = sum(x <= y for x, y in zip(ranks, ranks[1:]))
    state_tail = sum(b in STATE for _, b in order[-2:])
    return (monotonic, state_tail, -sum(abs(x-y) for x,y in zip(ranks, ranks[1:])))

def diagnostics(order):
    nouns = [a for a, _ in order]
    return {"order": [a+"/"+b for a,b in order], "score": score(order),
            "observation_to_state": [a for a,b in order if a not in STATE] + [b for a,b in order if b in STATE],
            "listiness_flags": {"repeated_template": True, "shared_predicate_risk": True,
                                "distinct_lexical_heads": len(set(nouns)) == len(nouns),
                                "flag": "high" if len(set(nouns)) == 6 else "medium"},
            "diagnostic_only": True}

def main():
    ranked = sorted((diagnostics(o) for o in itertools.permutations(PAIRS)), key=lambda d: d["score"], reverse=True)
    best = ranked[0]
    rng = random.Random(20261006)
    shuffled = list(PAIRS); rng.shuffle(shuffled)
    payload = {"experiment_id": "paragraph-family-observation-state-diagnostic-20261002",
      "method": "deterministic permutation ranking plus template/listiness flags",
      "family": [a+"/"+b for a,b in PAIRS], "best_order": best,
      "top_5": ranked[:5],
      "reader_packet": {"intact": {"id":"family-intact","order":best["order"],"purpose":"candidate narrative order"},
                        "shuffled": {"id":"family-shuffled","order":[a+"/"+b for a,b in shuffled],"seed":20261006,"purpose":"ordering/listiness control"},
                        "pair_reversed": {"id":"family-reversed","order":[b+"/"+a for a,b in PAIRS],"purpose":"surface reversal control"}},
      "interpretation": "The family is lexically distinctive but structurally list-like: six repeated short templates with no supplied participant/event bridges. Ranking is a narrative hypothesis, not a readability certificate.",
      "provenance": {"lm":False,"rlaif":False,"permutations":720,"seed":20261006},
      "next_repair": "Add authored observation-to-observation and observation-to-state bridges, then test intact versus shuffled packets with human readers."}
    OUT.write_text(json.dumps(payload, indent=2)+"\n")
    assert len(ranked)==720 and best["order"][0]=="trams/smart" and best["diagnostic_only"]
    print(json.dumps({"best":best["order"],"score":best["score"],"controls":3}))
if __name__ == "__main__": main()
