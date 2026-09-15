"""Bounded search over independently authored typed English templates.

This run differs from the earlier flat clause product: the inventory has
separate semantic slots (agent, action, patient, predicate, location,
modifier), and a closure is only considered when two *different* complete
surfaces have reverse character tapes.  The reverse side is segmented back
through the same typed inventory, so accidental word-order mirrors are not
silently accepted.  A failed run emits a concrete next operator: repair the
lowest-scoring boundary by replacing one lexical slot while preserving its
typed frame and observed-bigram constraints.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from collections import defaultdict
from itertools import product
from pathlib import Path

from nltk.corpus import brown
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.bigram import BigramModel

FUNCTION = set("a an the this that my our some many no one i me we you he she it they and or but if as of to in on at by for from with near is are was were be do does did can could will would have has had not".split())

# Complete ordinary surfaces, not fragments.  Slot roles are deliberately
# explicit so lexical semantics is part of search, not an afterthought.
FRAMES = {
    "transitive": ("det", "noun", "verb", "det", "noun"),
    "locative": ("det", "noun", "verb", "adp", "det", "noun"),
    "modified": ("det", "adj", "noun", "verb", "det", "noun"),
    "copular": ("det", "noun", "copula", "adj"),
    "agentive": ("pron", "verb", "det", "adj", "noun"),
}

def lexical_inventory(limit: int) -> dict[str, tuple[str, ...]]:
    tagged = defaultdict(set)
    for sent in brown.tagged_sents(tagset="universal"):
        for word, tag in sent:
            if word.isascii() and word.isalpha(): tagged[word.lower()].add(tag)
    ranked = [w for w in top_n_list("en", 40000)
              if w.isascii() and w.isalpha() and len(w) >= 3 and w in tagged
              and w != w[::-1] and zipf_frequency(w, "en") >= 3.4][:limit]
    def only(tag, excluded=()):
        return tuple(w for w in ranked if tag in tagged[w] and not any(x in tagged[w] for x in excluded)
                     and w not in FUNCTION)
    return {
        "det": tuple(w for w in "a an the this that my our some many no one".split() if w in tagged),
        "pron": tuple(w for w in "i me we you he she it they".split() if w in tagged),
        "adp": tuple(w for w in "by for in near on to with from at into over under".split() if w in tagged),
        "noun": only("NOUN", ("VERB", "ADJ", "ADV")),
        "verb": only("VERB", ("NOUN", "ADJ", "ADV")),
        "adj": only("ADJ", ("NOUN", "VERB", "ADV")),
        "copula": tuple(w for w in "is are was were be".split() if w in tagged),
    }

def surfaces(frame, inv, bg, cap):
    out = []
    def walk(i, words, score):
        if len(out) >= cap: return
        if i == len(frame):
            out.append((normalize_letters(" ".join(words)), tuple(words), score)); return
        for word in inv[frame[i]]:
            if word not in FUNCTION and word in words: continue
            if words and not bg.observed(words[-1], word): continue
            walk(i + 1, words + (word,), score + (bg.forward(words[-1], word) if words else 0) + zipf_frequency(word, "en") * .1)
    walk(0, (), 0.0)
    return out

def run(pool_limit=2200, surface_cap=50000):
    inv = lexical_inventory(pool_limit)
    vocab = {w for xs in inv.values() for w in xs}
    bg = BigramModel.from_file(str(ROOT / "data" / "count_2w.txt"), vocab=vocab)
    banks = {name: surfaces(frame, inv, bg, surface_cap) for name, frame in FRAMES.items()}
    reverse_index = defaultdict(list)
    for name, rows in banks.items():
        for tape, words, score in rows: reverse_index[tape].append((name, words, score))
    exact = []
    near = []
    for lname, rows in banks.items():
        for tape, left, ls in rows:
            # Reverse segmentation must be another complete typed surface.
            for rname, right, rs in reverse_index.get(tape[::-1], ()):
                if lname == rname or left == right: continue
                text = " ".join(left).capitalize() + "; " + " ".join(right) + "."
                checks = mechanical_admission_checks(text, min_letters=39, max_letters=180)
                row = {"text": text, "letters": len(normalize_letters(text)), "left_frame": lname, "right_frame": rname,
                       "mechanical_checks": checks, "score": ls + rs}
                exact.append(row)
    # The diagnostic is useful for the next operator even with no closure:
    # maximize reverse-prefix agreement between typed surfaces and record the
    # boundary at which a one-slot lexical substitution should be attempted.
    # Bound the repair diagnostic independently from closure search: compare
    # only same-length tapes and a deterministic 100-surface sample per frame.
    by_length = defaultdict(list)
    for n, rs in banks.items():
        for other, rw, _ in rs[:100]: by_length[len(other)].append((n, other, rw))
    for lname, rows in banks.items():
        for tape, words, score in rows[: min(1000, len(rows))]:
            pool = by_length[len(tape)]
            if not pool: continue
            best = max((sum(a == b for a, b in zip(tape, other)), other, n, rw)
                       for n, other, rw in pool)
            near.append({"left_frame": lname, "left": words, "right_frame": best[2], "matched_prefix": best[0], "left_letters": len(tape), "right_words": best[3]})
    unique = {x["text"]: x for x in exact}
    eligible = [x for x in unique.values() if all(x["mechanical_checks"].values()) and x["letters"] >= 39]
    return {"status": "complete", "config": {"pool_limit": pool_limit, "surface_cap": surface_cap, "frames": FRAMES, "observed_bigrams_hard": True, "minimum_letters": 39},
            "inventory_counts": {k: len(v) for k, v in inv.items()}, "surface_counts": {k: len(v) for k, v in banks.items()},
            "exact_closures": len(unique), "mechanically_eligible": eligible, "exact_records": list(unique.values()),
            "near_miss_sample": sorted(near, key=lambda x: -x["matched_prefix"])[:20],
            "repair_operator": "For each highest reverse-prefix near miss, replace exactly one lexical slot in the weaker boundary word while retaining its typed role and requiring both adjacent observed bigrams; re-run reverse segmentation before scoring.",
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "material": "Brown POS tags plus wordfreq lexical types and observed 2-word joins; no source sentence copied"},
            "reader_gate": "No programmatic result certifies readability; any eligible closure requires randomized blinded intact-prose and shuffled controls."}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--pool-limit", type=int, default=2200); ap.add_argument("--surface-cap", type=int, default=50000)
    a = ap.parse_args()
    if a.out.exists(): ap.error("refusing to overwrite output")
    result = run(a.pool_limit, a.surface_cap); a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"exact": result["exact_closures"], "eligible": len(result["mechanically_eligible"])}, indent=2))
