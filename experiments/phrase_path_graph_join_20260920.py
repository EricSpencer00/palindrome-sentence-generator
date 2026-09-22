"""Phrase-first graph join over observed transitions and typed clause parses.

Left clauses are generated as ordinary word paths from an observed bigram
graph.  The right side is not copied or rendered by reversing that clause: its
own typed parser matches words against the live reverse character obligation,
and each final surface receives an independent exact audit.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from experiments.forward_lexicalized_grammar_20260920 import independent_audit, letters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/phrase-path-graph-join-20260920.json"

PATTERNS = (
    ("DET", "N", "V", "NUM", "N"),
    ("DET", "N", "V", "DET", "N"),
    ("DET", "ADJ", "N", "V", "DET", "N"),
    ("PRON", "V", "DET", "N"),
    ("N", "V", "DET", "N"),
    ("PROPN", "V", "DET", "N"),
    ("DET", "N", "V", "PROPN"),
    ("DET", "N", "V", "PREP", "DET", "N"),
)


def load_pos(path=ROOT / "data/brown_pcfg_bank_20260920.json", limit=70):
    bank = json.loads(Path(path).read_text())["lexicon"]
    mapping = {"DET": "DET", "NOUN": "N", "VERB": "V", "ADJ": "ADJ",
               "PREP": "PREP", "PRON": "PRON"}
    pos = defaultdict(set)
    for source, tag in mapping.items():
        for row in bank[source][:limit]:
            word = row["word"].casefold()
            if re.fullmatch(r"[a-z]+", word):
                pos[word].add(tag)
    for word in "diana leon noel elba anna adam eve oscar ada otto".split():
        pos[word].add("PROPN")
    for word in "one two three four five six seven eight nine ten eleven twelve".split():
        pos[word].add("NUM")
    pos["aide"].add("N"); pos["rips"].add("V"); pos["memos"].add("N")
    pos["men"].add("N"); pos["inspire"].add("V")
    return {word: frozenset(tags) for word, tags in pos.items()}


def load_edges(path=ROOT / "data/count_2w.txt", edge_limit=300_000, fanout=40):
    edges = defaultdict(dict)
    rows = 0
    with Path(path).open(errors="ignore") as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) != 2:
                continue
            pair = fields[0].casefold().split()
            if len(pair) != 2 or not all(re.fullmatch(r"[a-z]+", w) for w in pair):
                continue
            try: count = int(fields[1])
            except ValueError: continue
            a, b = pair
            edges[a][b] = max(count, edges[a].get(b, 0))
            rows += 1
            if rows >= edge_limit: break
    nexts = {a: tuple(sorted(bs, key=lambda b: -bs[b])[:fanout]) for a, bs in edges.items()}
    return nexts, rows


def _nested(words):
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            if i == 0 and j == len(words): continue
            tape = letters(" ".join(words[i:j]))
            if len(tape) > 1 and tape == tape[::-1]: return True
    return False


def _parse_obligation(tape, pattern, by_pos, pos, chosen):
    if pos == len(pattern):
        return [tuple(chosen)] if len(tape) == 0 else []
    tag = pattern[pos]
    out = []
    for word in by_pos.get(tag, ()):
        w = letters(word)
        if tape.startswith(w):
            out.extend(_parse_obligation(tape[len(w):], pattern, by_pos, pos + 1, chosen + [word]))
    return out


def run(*, pos_limit=70, edge_limit=300_000, fanout=40, max_left_paths=2_000_000):
    pos = load_pos(limit=pos_limit)
    nexts, rows = load_edges(edge_limit=edge_limit, fanout=fanout)
    by_pos = {tag: tuple(sorted((word for word, tags in pos.items() if tag in tags),
                                key=lambda w: (len(w), w)))
              for tag in {tag for tags in pos.values() for tag in tags}}
    candidates = {}
    generated = 0
    left_paths = 0
    left_by_tag = by_pos

    def walk(path, pattern):
        nonlocal generated, left_paths
        if left_paths >= max_left_paths: return
        i = len(path)
        if i == len(pattern):
            left_paths += 1
            tape = letters(" ".join(path))
            reverse_tape = tape[::-1]
            for right_pattern in PATTERNS:
                for right in _parse_obligation(reverse_tape, right_pattern, left_by_tag, 0, []):
                    words = tuple(path) + tuple(right)
                    if len(tape) < 39 or len(words) != len(set(words)) or _nested(words): continue
                    text = " ".join(path) + "; " + " ".join(right) + "."
                    audit = independent_audit(text)
                    if audit["exact"] and audit["letters"] >= 39:
                        candidates[text] = {"length": audit["letters"], "rendered": text,
                                            "words": words, "audit": audit,
                                            "left_pattern": pattern, "right_pattern": right_pattern,
                                            "provenance": {"observed_word_paths": True,
                                                           "typed_right_parser": True,
                                                           "independent_character_audit": True,
                                                           "finished_tape_reversal": False,
                                                           "post_hoc_repair": False,
                                                           "word_order_mirroring": False,
                                                           "catalogue_text": False}}
            return
        tag = pattern[i]
        choices = left_by_tag.get(tag, ()) if not path else tuple(w for w in nexts.get(path[-1], ())
                                                                    if tag in pos.get(w, ()))
        for word in choices:
            if word in path: continue
            walk(path + (word,), pattern)

    for pattern in PATTERNS:
        walk((), pattern)
        if left_paths >= max_left_paths: break
    result = {"paths": sorted(candidates.values(), key=lambda r: (-r["length"], r["rendered"])),
              "stats": {"observed_rows": rows, "left_paths": left_paths,
                        "exact": len(candidates), "status": "SAT" if candidates else "UNSAT"},
              "experiment_id": "phrase-path-graph-join-20260920",
              "provenance": {"method": "observed transition graph left paths joined to typed reverse obligation parser",
                             "typed_right_parser": True,
                             "pos_limit": pos_limit, "edge_limit": edge_limit, "fanout": fanout,
                             "max_left_paths": max_left_paths, "reader_gate": "closed pending blinded intact-versus-shuffled reading",
                             "next_construction": "if empty, abandon observed-corpus path family for authored scene topology"}}
    return result


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for row in result["paths"][:20]: print(row["rendered"])
