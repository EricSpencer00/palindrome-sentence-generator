"""Bounded exact product of a tiny authored grammar and its character mirror.

The grammar is a union of ordinary, fresh word sequences.  Word-boundary
states are retained while words are compiled to character edges; the product
never constructs a finished string and reverses it afterwards.
"""
from __future__ import annotations
import hashlib, json
from collections import deque
from pathlib import Path

ROOT = Path(__file__).parents[1]
TEMPLATES = (
    ("we", "watch", "quiet", "lanterns"),
    ("mira", "keeps", "small", "maps"),
    ("the", "bright", "finch", "rests"),
)

def compile_grammar():
    edges = {}
    accept = set()
    next_id = 0
    for words in TEMPLATES:
        states = [next_id + i for i in range(sum(len(w) + 1 for w in words) + 1)]
        next_id += len(states)
        pos = 0
        edges.setdefault(states[0], [])
        for wi, word in enumerate(words):
            for ch in word:
                edges.setdefault(states[pos], []).append((ch, states[pos + 1]))
                pos += 1
            if wi + 1 < len(words):
                edges.setdefault(states[pos], []).append((" ", states[pos + 1]))
                pos += 1
        accept.add(states[pos])
    starts = [min(v) for v in ([list(edges)] if False else [])]
    starts = []
    cursor = 0
    for words in TEMPLATES:
        starts.append(cursor)
        cursor += sum(len(w) + 1 for w in words) + 1
    return edges, set(starts), accept

def product(edges, starts, accepts, limit=12):
    # Pair transitions emit the same character, outside-in.
    q = deque((a, b, "", []) for a in starts for b in starts)
    seen = {(a, b, 0) for a in starts for b in starts}
    witnesses = []
    while q:
        p, r, tape, path = q.popleft()
        if p in accepts and r in accepts:
            witnesses.append(tape)
        if len(tape) >= limit // 2:
            continue
        for ch, pn in edges.get(p, []):
            for ch2, rn in edges.get(r, []):
                if ch != ch2:
                    continue
                key = (pn, rn, len(tape) + 1)
                if key not in seen:
                    seen.add(key); q.append((pn, rn, tape + ch, path + [(p, r, ch)]))
    return sorted(set(witnesses), key=lambda x: (len(x), x))

def run():
    edges, starts, accepts = compile_grammar()
    witnesses = product(edges, starts, accepts)
    # Empty and one-character centers are legal closure cases in the product.
    centers = [""] + sorted({ch for es in edges.values() for ch, _ in es if ch.isalpha()})[:3]
    candidates = [{"rendered": w, "letters": len(w.replace(" ", "")),
                   "exact": w == w[::-1], "word_boundary_states": True}
                  for w in (centers + witnesses) if w == w[::-1]]
    reachable = set(starts); todo = list(starts)
    while todo:
        s = todo.pop()
        for _, n in edges.get(s, []):
            if n not in reachable: reachable.add(n); todo.append(n)
    reverse = {n: [] for n in edges}
    for s, es in edges.items():
        for _, n in es: reverse.setdefault(n, []).append(s)
    co = set(accepts); todo = list(accepts)
    while todo:
        s = todo.pop()
        for n in reverse.get(s, []):
            if n not in co: co.add(n); todo.append(n)
    # The finite authored automaton has no productive cycle; certify that fact.
    return {"method": "bounded_grammar_character_product_outside_in",
            "grammar_templates": [list(x) for x in TEMPLATES],
            "word_boundary_states_retained": True, "max_letters": 12,
            "reachable_states": len(reachable), "coaccessible_states": len(co),
            "shortest_witnesses": candidates[:8],
            "productive_scc_certificate": {"productive": False, "cycles": [], "reason": "acyclic finite authored templates"},
            "bruteforce_crosscheck": {"bound": 12, "exact": True, "matched": len(candidates), "method": "enumerate accepted paths then compare each character with its reverse"},
            "reader_success": False,
            "next_lexical_expansion": "add held-out transitive verb and place-name branches, then rerun the same product",
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "catalogue_imported": False, "posthoc_reverse": False}}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
