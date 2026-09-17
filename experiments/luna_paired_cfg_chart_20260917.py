"""Paired typed-CFG chart search with character obligations.

This is deliberately not a larger beam or a completed-sentence filter.  Two
independent typed derivations are expanded one production at a time.  Their
terminal characters are consumed from opposite ends by a small chart state;
states whose live residual cannot be completed by the remaining lexical
domains are discarded before either sentence is rendered.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks

SLOTS = ("DET", "SUBJ", "VERB", "OBJ", "PREP", "PLACE")
DOMAINS = {
    "DET": ("the", "a", "this", "our", "one"),
    "SUBJ": ("baker", "doctor", "farmer", "guard", "teacher", "writer"),
    "VERB": ("carries", "draws", "helps", "marks", "reads", "sends"),
    "OBJ": ("letter", "message", "map", "memo", "parcel", "story"),
    "PREP": ("at", "by", "in", "near", "on"),
    "PLACE": ("home", "school", "town", "garden", "market", "office"),
}
# Semantic valency is part of the grammar, not a post-hoc score.
VALENCY = {"carries": "TRANS", "draws": "TRANS", "helps": "TRANS",
           "marks": "TRANS", "reads": "TRANS", "sends": "TRANS"}

def emit(word: str) -> str:
    return normalize_letters(word)

def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def compatible(prefix: str, suffix: str) -> bool:
    """Necessary condition: known outer characters agree."""
    n = min(len(prefix), len(suffix))
    return prefix[:n] == suffix[::-1][:n]

def search(limit: int = 250_000):
    # Chart key is (slot index, unmatched side, unmatched residual, valency).
    # The residual is the exact character debt between the two partial tapes.
    chart = {(0, "", "", "TRANS"): ("", "", ())}
    counts = [1]
    pruned = defaultdict(int)
    for i, slot in enumerate(SLOTS):
        rslot = SLOTS[-1 - i]
        nxt = {}
        for (idx, side, debt, val), (left, right, words) in chart.items():
            for lw in DOMAINS[slot]:
                for rw in DOMAINS[rslot]:
                    if slot == "VERB" and VALENCY.get(lw) != val:
                        continue
                    # Independent lexicalization: no right word is copied.
                    a, b = emit(lw), emit(rw)[::-1]
                    if side == "L":
                        x, y = debt + a, b
                    elif side == "R":
                        x, y = a, debt + b
                    else:
                        x, y = a, b
                    k = min(len(x), len(y))
                    if x[:k] != y[:k]:
                        pruned["mirrored_character_conflict"] += 1
                        continue
                    if len(x) > len(y): ns, nd = "L", x[k:]
                    elif len(y) > len(x): ns, nd = "R", y[k:]
                    else: ns, nd = "", ""
                    # Chart dominance: retain one witness for each live debt.
                    key = (i + 1, ns, nd, val)
                    if key not in nxt:
                        nxt[key] = (left + " " + lw, right + " " + rw,
                                    words + ((lw, rw),))
                    if len(nxt) >= limit:
                        break
                if len(nxt) >= limit: break
            if len(nxt) >= limit: break
        chart = nxt
        counts.append(len(chart))
        if not chart: break
    candidates = []
    for (idx, side, debt, val), (left, right, words) in chart.items():
        if idx == len(SLOTS) and not debt:
            text = left.strip() + " " + " ".join(reversed(right.strip().split()))
            audit = mechanical_admission_checks(text)
            candidates.append({"text": text, "audit": audit,
                               "letters": len(emit(text)), "words": words,
                               "provenance": "paired_typed_cfg_chart"})
    return {"chart_counts": counts, "terminal_candidates": candidates,
            "pruned": dict(pruned), "limit": limit,
            "domain_sizes": {k: len(v) for k, v in DOMAINS.items()},
            "grammar": list(SLOTS)}

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=250000)
    ap.add_argument("--out", type=Path, default=Path("runs/luna-paired-cfg-chart-20260917.json"))
    args = ap.parse_args(); result = search(args.limit)
    rendered = []
    for c in result["terminal_candidates"]:
        c["independent_tape"] = normalize_letters(c["text"]) == normalize_letters(c["text"])[::-1]
        rendered.append(c)
    result["terminal_candidates"] = rendered
    result["run_sha256"] = digest(result)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"chart_counts": result["chart_counts"], "terminals": len(rendered), "pruned": result["pruned"]}, indent=2))

if __name__ == "__main__": main()
