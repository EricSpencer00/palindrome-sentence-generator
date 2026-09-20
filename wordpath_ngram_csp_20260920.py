"""Open word-path lattice intersected with live palindrome character cells.

Unlike the fixed two-clause grammar, this lane lets each side choose a
variable-length observed word path.  A path edge must be an observed adjacent
word transition; the two paths consume opposite character residuals before
the next edge is chosen.  This is a search-space constructor, not a scorer.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from bilateral_grammar_csp_20260920 import _consume
from forward_lexicalized_grammar_20260920 import admission_ok, independent_audit, letters

ROOT = Path(__file__).resolve().parent


def load_wordpath_lattice(edge_path=ROOT / "data/count_2w.txt", edge_limit=200_000, vocab_limit=3_000):
    nexts = defaultdict(set)
    prevs = defaultdict(set)
    rows = 0
    with Path(edge_path).open(errors="ignore") as handle:
        for line in handle:
            pair = line.split("\t", 1)[0].casefold().split()
            if len(pair) != 2 or not all(re.fullmatch(r"[a-z]+", word) for word in pair):
                continue
            a, b = pair
            nexts[a].add(b)
            prevs[b].add(a)
            rows += 1
            if rows >= edge_limit:
                break
    vocab = sorted(set(nexts) | set(prevs), key=lambda word: (-len(nexts[word]) - len(prevs[word]), word))[:vocab_limit]
    vocab_set = set(vocab)
    nexts = {word: tuple(sorted(nxt & vocab_set)) for word, nxt in nexts.items() if word in vocab_set}
    prevs = {word: tuple(sorted(prv & vocab_set)) for word, prv in prevs.items() if word in vocab_set}
    return tuple(vocab), nexts, prevs, rows


def wordpath_csp(vocab, nexts, prevs, max_words=12, min_side_words=3, max_nodes=500_000):
    starts = tuple(vocab)
    found = []
    stats = {"nodes": 0, "pruned_character": 0, "pruned_repeat": 0, "complete": 0}
    by_last = defaultdict(tuple)
    grouped = defaultdict(list)
    for word in vocab:
        grouped[word[-1]].append(word)
    by_last = {key: tuple(value) for key, value in grouped.items()}

    def search(left, right_rev, lres="", rres=""):
        if stats["nodes"] >= max_nodes:
            return
        stats["nodes"] += 1
        if lres == rres == "" and len(left) >= min_side_words and len(right_rev) >= min_side_words:
            words = left + list(reversed(right_rev))
            if admission_ok(words):
                text = " ".join(words)
                audit = independent_audit(text)
                if audit["exact"] and audit["letters"] >= 39:
                    found.append({"length": audit["letters"], "rendered": " ".join(left) + "; " + " ".join(reversed(right_rev)) + ".",
                                  "words": words, "audit": audit,
                                  "provenance": {"observed_word_edges": True, "variable_side_lengths": True,
                                                 "live_character_residual": True, "candidate_reranking": False,
                                                 "finished_tape_reversal": False, "post_hoc_repair": False,
                                                 "word_order_mirroring": False, "catalogue_text": False}})
        if len(left) + len(right_rev) >= max_words:
            return
        left_choices = starts if not left else nexts.get(left[-1], ())
        right_choices = starts if not right_rev else prevs.get(right_rev[-1], ())
        if not lres and not rres:
            # The next left character must equal the next character exposed
            # from the right edge (the right word's final character).
            right_choices_by_char = by_last
        else:
            right_choices_by_char = None
        # The first exposed characters are the strongest cheap filter.  Pair
        # only words whose exposed character can match when no residual exists.
        for a in left_choices:
            candidate_right = (tuple(b for b in right_choices_by_char.get(a[0], ()) if b in right_choices)
                               if right_choices_by_char is not None else right_choices)
            for b in candidate_right:
                if a in left or b in left or a in right_rev or b in right_rev:
                    stats["pruned_repeat"] += 1
                    continue
                residual = _consume(lres + letters(a), rres + letters(b)[::-1])
                if residual is None:
                    stats["pruned_character"] += 1
                    continue
                search(left + (a,), right_rev + (b,), *residual)

    # Start with every exposed outer word pair, then grow variable paths.
    search((), ())
    stats["complete"] = len(found)
    stats["status"] = "timeout" if stats["nodes"] >= max_nodes else ("SAT" if found else "UNSAT")
    unique = {row["rendered"]: row for row in found}
    return {"paths": sorted(unique.values(), key=lambda row: (-row["length"], row["rendered"])), "stats": stats,
            "provenance": {"method": "observed word-path lattice with live bilateral character residual",
                           "candidate_reranking": False,
                           "next_construction": "apply an independently authored clause/valency parser to exact survivors before reader testing"}}


if __name__ == "__main__":
    vocab, nexts, prevs, rows = load_wordpath_lattice()
    result = wordpath_csp(vocab, nexts, prevs)
    result["provenance"].update({"vocab": len(vocab), "observed_rows": rows})
    print(json.dumps(result["stats"]))
