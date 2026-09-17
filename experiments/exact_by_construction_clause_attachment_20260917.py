"""Repair of the optional-slot product: attach a second complete clause.

The preceding lane exhausted a single clause automaton before reaching a
reader-worthy closure.  This repair composes two independently compiled
ordinary clauses with an explicit ``and`` edge, preserving exact character
matching during construction and complete-clause parsing on both sides.
"""
from __future__ import annotations

import hashlib
import json
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "exact-by-construction-clause-attachment-20260917.json"

_SPEC = importlib.util.spec_from_file_location(
    "exact_by_construction_clause_product_20260917",
    ROOT / "experiments/exact_by_construction_clause_product_20260917.py",
)
assert _SPEC and _SPEC.loader
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)


def attached_grammar() -> base.Grammar:
    first = base.build_clause_grammar()
    max_node = max(max(edge.source, edge.target) for edge in first.edges)
    offset = max_node + 1
    edges = list(first.edges)
    edges.extend(
        base.Edge(edge.source + offset, edge.target + offset, edge.char,
                  edge.completed_word, edge.role)
        for edge in first.edges
    )
    next_node = [max(max(edge.source, edge.target) for edge in edges) + 1]
    # The conjunction is a real lexical edge, so it contributes characters and
    # cannot hide an epsilon seam.
    base._word_edges(edges, next_node, first.end, offset, "and", "conjunction")
    role_words = {**first.role_words, "conjunction": ("and",)}
    role_transitions = list(first.role_transitions) + [(first.end, "conjunction", offset)]
    return base.Grammar(first.start, first.end + offset, tuple(edges), role_words,
                        tuple(role_transitions))


def parse_attached(words: list[str], grammar: base.Grammar) -> dict:
    if "and" not in words:
        parsed = base.independent_parse(grammar, words)
        return {"ok": parsed["ok"], "clauses": [parsed]}
    split = words.index("and")
    left = base.independent_parse(base.build_clause_grammar(), words[:split])
    right = base.independent_parse(base.build_clause_grammar(), words[split + 1:])
    return {"ok": left["ok"] and right["ok"], "clauses": [left, right]}


def run(max_states: int = 500_000) -> dict:
    grammar = attached_grammar()
    product = base.exact_product(grammar, max_states=max_states)
    rows = []
    admitted = []
    for record in product["records"]:
        text = record["text"].capitalize() + "."
        audit = base.independent_audit(text)
        parse = parse_attached(record["words"], grammar)
        shortcut = base.anti_shortcut(record["words"])
        accepted = (40 <= audit["letters"] <= 140 and audit["exact"] and parse["ok"]
                    and not shortcut["word_order_mirror"]
                    and not shortcut["self_palindromic_spans"]
                    and not shortcut["repeated_words"])
        row = {"rendered": text, "words": record["words"], "roles": record["roles"],
               "letters": audit["letters"], "independent_audit": audit,
               "independent_parse": parse, "anti_shortcut": shortcut,
               "mechanically_admitted": accepted,
               "reader_status": "human-unreviewed; programmatic checks do not certify readability",
               "provenance": {"author_authored_lexicon": True, "catalogue_seed": False}}
        rows.append(row)
        if accepted:
            admitted.append(row)
    result = {
        "status": "completed_no_admitted_closure" if not admitted else "completed_admitted_closure",
        "method": "exact_by_construction_optional_slot_clause_attachment",
        "search": {"states": product["states"], "max_states": max_states,
                   "truncated": product["truncated"], "rlaif_per_candidate": False,
                   "independent_clause_attachment": True},
        "exact_candidates": rows, "mechanically_admitted": admitted,
        "reader_facing_next_test": (
            "Only an admitted intact closure can enter the randomized blinded reader package; "
            "if none appears, expand the attachment with a typed relative-clause edge."),
        "next_repair": {"operator": "typed_relative_clause_attachment",
                        "target": "add a second complete clause without relaxing character equality",
                        "reason": "the conjunction attachment preserves exact construction but exhausted before a closure"},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "parent_lane": "exact-by-construction-optional-slot-clause-product-20260917",
                       "seed_used_in_output": False},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
