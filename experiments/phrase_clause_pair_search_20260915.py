"""Generate complete clauses while matching reverse character tapes.

This is the small constructive branch missing from the earlier residual runs:
lexical role choices and exact reversal are joined at the *clause* boundary,
while observed word joins keep the resulting clauses locally readable.  It
does not copy corpus sentences; Brown is used only for POS tags and bigram
counts.  Every exact result still goes through mechanical admission and a
human reader gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

from nltk.corpus import brown
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.bigram import BigramModel

FUNCTION = frozenset("a an the this that these those my our some many no one i me we us you he him she her it they them who which and or but if as while when after before because though of to in on at by for from with without near during is are was were be been do does did can could will would may might should have has had not".split())

SHAPES = (
    ("det", "noun", "verb", "det", "noun"),
    ("pron", "verb", "det", "noun"),
    ("name", "verb", "det", "noun"),
    ("det", "adj", "noun", "verb", "det", "noun"),
    ("pron", "verb", "adp", "det", "noun"),
    ("det", "noun", "verb", "adp", "det", "noun"),
    ("adv", "pron", "verb", "det", "noun"),
    ("det", "noun", "verb", "det", "adj", "noun"),
    ("name", "verb", "name"),
    ("det", "noun", "copula", "adj"),
)


def table() -> dict[str, frozenset[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for sentence in brown.tagged_sents(tagset="universal"):
        for word, tag in sentence:
            if word.isascii() and word.isalpha():
                out[word.casefold()].add(tag)
    return {w: frozenset(tags) for w, tags in out.items()}


def pools(tags: dict[str, frozenset[str]], size: int) -> dict[str, tuple[str, ...]]:
    ranked = [w for w in top_n_list("en", 30000)
              if w.isascii() and w.isalpha() and w in tags and len(w) >= 2
              and w != w[::-1] and zipf_frequency(w, "en") >= 3.2][:size]
    det = tuple(w for w in "a an the this that these those my our some many no one".split()
                if w in tags and w != w[::-1])
    pron = tuple(w for w in "i me we us you he him she her it they them".split()
                 if w in tags and w != w[::-1])
    adp = tuple(w for w in "by for in near on to with from at into over under".split()
                if w in tags and w != w[::-1])
    # Keep noun/verb/adjective roles separate; ambiguous items can turn a
    # locally fluent tape into a syntactic category error.
    noun = tuple(w for w in ranked if "NOUN" in tags[w] and not ({"VERB", "ADJ", "ADV"} & set(tags[w])) and w not in FUNCTION)
    verb = tuple(w for w in ranked if "VERB" in tags[w] and not ({"NOUN", "ADJ", "ADV"} & set(tags[w])) and w not in FUNCTION)
    adj = tuple(w for w in ranked if "ADJ" in tags[w] and not ({"NOUN", "VERB", "ADV"} & set(tags[w])) and w not in FUNCTION)
    adv = tuple(w for w in ranked if "ADV" in tags[w] and not ({"NOUN", "VERB", "ADJ"} & set(tags[w])) and w not in FUNCTION)
    name = tuple(w for w in ranked if w in {"alice", "alan", "ari", "diana", "eva", "ira", "liam", "mia", "nadia", "nora", "noel", "leon", "anna", "nina", "sara", "maya", "olivia"})
    copula = tuple(w for w in "is are was were be".split() if w in tags)
    return {"det": det, "pron": pron, "adp": adp, "noun": noun, "verb": verb,
            "adj": adj, "adv": adv, "name": name, "copula": copula}


def clause_rows(shape, p, bg, cap=250_000):
    rows = []
    def walk(i, words, score):
        if len(rows) >= cap:
            return
        if i == len(shape):
            tape = "".join(words)
            rows.append((tape, tuple(words), score)); return
        role = shape[i]
        for word in p[role]:
            if word not in FUNCTION and word in words:
                continue
            gain = 0.0 if not words else bg.forward(words[-1], word)
            # Requiring attested joins is a hard construction condition here;
            # punctuation may later split complete clauses but cannot repair a
            # bad interior join.
            if words and not bg.observed(words[-1], word):
                continue
            walk(i + 1, words + (word,), score + gain + 0.12 * zipf_frequency(word, "en"))
    walk(0, (), 0.0)
    return rows


def run(pool_size: int = 5000, clause_cap: int = 250_000):
    tags = table(); p = pools(tags, pool_size)
    vocab = set(w for words in p.values() for w in words)
    bg = BigramModel.from_file(str(ROOT / "data" / "count_2w.txt"), vocab=vocab)
    banks = {shape: clause_rows(shape, p, bg, clause_cap) for shape in SHAPES}
    indices: dict[str, list[tuple[tuple[str, ...], float]]] = defaultdict(list)
    for shape, rows in banks.items():
        for tape, words, score in rows:
            indices[tape].append((words, score))
    out = []
    for left_shape, rows in banks.items():
        for tape, left_words, left_score in rows:
            for right_words, right_score in indices.get(tape[::-1], ()):
                words = left_words + right_words
                text = " ".join(left_words).capitalize() + "; " + " ".join(right_words) + "."
                norm = normalize_letters(text)
                if len(norm) < 39 or norm != norm[::-1]:
                    continue
                checks = mechanical_admission_checks(text, min_letters=39, max_letters=180)
                out.append({"text": text, "letters": len(norm), "left_shape": left_shape,
                            "left_words": left_words, "right_words": right_words,
                            "score": left_score + right_score, "mechanical_checks": checks,
                            "mechanically_eligible": all(checks.values())})
    unique = {r["text"]: r for r in out}
    return {"status": "phrase_clause_pair_search_complete",
            "config": {"pool_size": pool_size, "clause_cap": clause_cap,
                       "shape_count": len(SHAPES), "minimum_letters": 39,
                       "observed_bigrams_hard": True},
            "pool_counts": {k: len(v) for k, v in p.items()},
            "clause_counts": {str(shape): len(rows) for shape, rows in banks.items()},
            "exact_closures": len(out), "unique_exact_closures": len(unique),
            "mechanically_eligible": sorted((r for r in unique.values() if r["mechanically_eligible"]), key=lambda r: (-r["letters"], -r["score"])),
            "exact_records": sorted(unique.values(), key=lambda r: (-r["letters"], -r["score"])),
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "material": "wordfreq-ranked lexical types with Brown POS and observed bigram joins; no sentence copied"},
            "reader_gate": "A mechanically eligible closure is only a reader-study candidate; use randomized blinded intact prose and shuffled controls."}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pool-size", type=int, default=5000); ap.add_argument("--clause-cap", type=int, default=250000)
    args = ap.parse_args()
    if args.out.exists(): ap.error("refusing to overwrite output")
    result = run(args.pool_size, args.clause_cap); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"exact": result["unique_exact_closures"], "eligible": len(result["mechanically_eligible"])}, indent=2))
