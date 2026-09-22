"""Memoized weighted-CFG intersection with live palindrome obligations.

This is a chart, not a candidate repair pass: a hyperedge expands one CFG
nonterminal, while terminal edges consume the currently exposed characters on
the two ordinary-order frontiers.  Equivalent (grammar-stack, residual,
semantic-role) states are merged before lexical successors are explored.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.bilateral_grammar_csp_20260920 import _consume
from experiments.forward_lexicalized_grammar_20260920 import Word, independent_audit, letters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/cfg-hypergraph-obligation-20260920.json"

GRAMMAR = {
    "CLAUSE": (("SUBJ", "V", "OBJ"), ("SUBJ", "V", "PP")),
    "SUBJ": (("DET", "N"), ("PROPN",), ("PRON",)),
    "OBJ": (("DET", "N"), ("N",), ("PROPN",)),
    "PP": (("PREP", "DET", "N"),),
}


@dataclass(frozen=True)
class Item:
    left_stack: tuple[str, ...]
    right_stack: tuple[str, ...]
    left: tuple[str, ...]
    right: tuple[str, ...]  # ordinary order, accumulated from the right
    residual_l: str
    residual_r: str
    role: str
    weight: float


def load_lexicon(path=ROOT / "data/brown_pcfg_bank_20260920.json", limit=24):
    bank = json.loads(Path(path).read_text())["lexicon"]
    mapping = {"DET": "DET", "NOUN": "N", "VERB": "V", "PREP": "PREP", "PRON": "PRON"}
    rows = []
    for source, pos in mapping.items():
        for row in bank[source][:limit]:
            text = row["word"].casefold()
            if re.fullmatch(r"[a-z]+", text):
                rows.append(Word(text, pos))
    rows += [Word(x, "PROPN") for x in "diana leon noel elba anna adam".split()]
    unique = {(x.text, x.pos): x for x in rows}
    return tuple(unique.values())


def _expand(stack, grammar):
    if not stack or stack[0] not in grammar:
        return ()
    return tuple(tuple(prod) + stack[1:] for prod in grammar[stack[0]])


def search(lexicon, *, max_words=10, max_items=120_000):
    by_pos = {}
    for word in lexicon:
        by_pos.setdefault(word.pos, []).append(word)
    # A stable lexical weight orders hyperedges only; it is not a readability
    # certificate and is intentionally absent from candidate admission.
    common = {"the": 8.0, "a": 7.0, "an": 6.5, "some": 6.0, "men": 5.0}
    for rows in by_pos.values():
        rows.sort(key=lambda w: (-common.get(w.text, 1.0), w.text))
    chart = {}
    agenda = [Item(("CLAUSE",), ("CLAUSE",), (), (), "", "", "clause", 0.0)]
    stats = {"popped": 0, "unique_items": 0, "merged": 0, "grammar_hyperedges": 0,
             "terminal_edges": 0, "char_prunes": 0, "repeat_prunes": 0, "complete": 0}
    found = {}
    while agenda and stats["popped"] < max_items:
        item = agenda.pop()
        stats["popped"] += 1
        key = (item.left_stack, item.right_stack, item.left, item.right,
               item.residual_l, item.residual_r, item.role)
        old = chart.get(key)
        if old is not None:
            stats["merged"] += 1
            if old.weight >= item.weight:
                continue
        else:
            stats["unique_items"] += 1
        chart[key] = item
        if not item.left_stack and not item.right_stack:
            stats["complete"] += 1
            if item.residual_l or item.residual_r or len(item.left) + len(item.right) < 4:
                continue
            text = " ".join(item.left + item.right)
            audit = independent_audit(text)
            if audit["exact"] and audit["letters"] >= 39:
                found[text] = {"rendered": text + ".", "length": audit["letters"],
                               "audit": audit, "provenance": {"method": "memoized CFG hypergraph",
                               "ordinary_order_frontiers": True, "finished_tape_reversal": False,
                               "post_hoc_repair": False, "catalogue_text": False}}
            continue
        if len(item.left) + len(item.right) >= max_words:
            continue
        # Hyperedges expand one grammar symbol on either frontier. Right stack
        # is reversed so its last symbol is the next ordinary-order terminal.
        if item.left_stack and item.left_stack[0] in GRAMMAR:
            for prod in _expand(item.left_stack, GRAMMAR):
                stats["grammar_hyperedges"] += 1
                agenda.append(Item(prod, item.right_stack, item.left, item.right,
                                   item.residual_l, item.residual_r, item.role, item.weight))
            continue
        if item.right_stack and item.right_stack[-1] in GRAMMAR:
            for prod in GRAMMAR[item.right_stack[-1]]:
                stats["grammar_hyperedges"] += 1
                agenda.append(Item(item.left_stack, item.right_stack[:-1] + tuple(prod), item.left,
                                   item.right, item.residual_l, item.residual_r, item.role, item.weight))
            continue
        lefts = by_pos.get(item.left_stack[0], ()) if item.left_stack else (None,)
        rights = by_pos.get(item.right_stack[-1], ()) if item.right_stack else (None,)
        for lw in lefts:
            for rw in rights:
                chosen = item.left + item.right
                if (lw and lw.text in chosen) or (rw and rw.text in chosen):
                    stats["repeat_prunes"] += 1
                    continue
                ls = item.residual_l + (letters(lw.text) if lw else "")
                rs = item.residual_r + (letters(rw.text)[::-1] if rw else "")
                residual = _consume(ls, rs)
                if residual is None:
                    stats["char_prunes"] += 1
                    continue
                stats["terminal_edges"] += 1
                agenda.append(Item(item.left_stack[1:] if lw else item.left_stack,
                                   item.right_stack[:-1] if rw else item.right_stack,
                                   item.left + ((lw.text,) if lw else ()),
                                   ((rw.text,) if rw else ()) + item.right,
                                   residual[0], residual[1], item.role,
                                   item.weight + (common.get(lw.text, 1.0) if lw else 0) +
                                   (common.get(rw.text, 1.0) if rw else 0)))
    stats["status"] = "SAT" if found else ("LIMIT" if agenda else "UNSAT")
    return {"paths": sorted(found.values(), key=lambda x: -x["length"]), "stats": stats}


def run(*, lexicon_limit=24, max_words=10, max_items=120_000):
    result = search(load_lexicon(limit=lexicon_limit), max_words=max_words, max_items=max_items)
    result["experiment_id"] = "cfg-hypergraph-obligation-20260920"
    result["provenance"] = {"method": "weighted CFG hypergraph intersection with memoized character obligations",
                             "lexicon_limit": lexicon_limit, "max_words": max_words,
                             "max_items": max_items, "reader_gate": "closed",
                             "next_repair": "add a semantic frame hyperedge, not a lexical resweep"}
    return result


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
    for row in result["paths"][:10]:
        print(row["rendered"])
