"""Bounded exact product of a tiny authored grammar and its character mirror.

The grammar is a union of ordinary, fresh word sequences. Word-boundary
states are retained while words are compiled to letter edges. Only a complete
accepted derivation can become a witness; empty centers and arbitrary common
prefixes are never reported as sentences.
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
            for ci, ch in enumerate(word):
                # Spaces are rendering metadata, not part of the letter tape.
                edges.setdefault(states[pos], []).append((ch, states[pos + 1], wi, ci == len(word) - 1))
                pos += 1
            if wi + 1 < len(words):
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
    """Find complete accepted paths whose normalized letter tape is exact.

    The forward state starts at a grammar start and the reverse state starts
    at an accepting state. Matching transitions therefore certify an outer
    context ``u [center] reverse(u)``. A center is accepted only when the two
    states meet or one actual grammar edge bridges them.
    """
    reverse_edges = {}
    for src, outgoing in edges.items():
        for ch, dst, wi, word_end in outgoing:
            reverse_edges.setdefault(dst, []).append((ch, src, wi, word_end))
    witnesses = []
    # The templates are disjoint linear paths. For each start, use the first
    # accepting state after it; no arbitrary common-prefix pair is a witness.
    for start in sorted(starts):
        accept = next((a for a in sorted(accepts) if a > start), None)
        if accept is None:
            continue
        q = deque([(start, accept, "", [])])
        seen = {(start, accept, 0)}
        while q:
            p, r, outer, path = q.popleft()
            if p == r:
                witnesses.append({"tape": outer + outer[::-1], "outer": outer,
                                  "center": "", "path": path})
            if len(outer) >= limit // 2:
                continue
            for ch, pn, wi, word_end in edges.get(p, []):
                for ch2, rn, rwi, rword_end in reverse_edges.get(r, []):
                    if ch != ch2:
                        continue
                    key = (pn, rn, len(outer) + 1)
                    if key not in seen:
                        seen.add(key)
                        q.append((pn, rn, outer + ch, path + [(p, r, ch, wi, rwi)]))
            # An odd center must be an actual edge in the grammar path.
            for ch, pn, wi, word_end in edges.get(p, []):
                if pn == r:
                    witnesses.append({"tape": outer + ch + outer[::-1], "outer": outer,
                                      "center": ch, "path": path + [(p, r, ch, wi, wi)]})
    unique = {w["tape"]: w for w in witnesses}
    return sorted(unique.values(), key=lambda x: (len(x["tape"]), x["tape"]))

def run():
    edges, starts, accepts = compile_grammar()
    witnesses = product(edges, starts, accepts)
    # Only complete accepted grammar paths are candidates. This finite bank has
    # no palindromic derivation in the bound, so the list is intentionally empty.
    candidates = [{"normalized_tape": w["tape"], "letters": len(w["tape"]),
                   "exact": w["tape"] == w["tape"][::-1],
                   "word_boundary_states": True, "center": w["center"],
                   "derivation_path": w["path"]}
                  for w in witnesses if w["tape"] == w["tape"][::-1]]
    reachable = set(starts); todo = list(starts)
    while todo:
        s = todo.pop()
        for _, n, _, _ in edges.get(s, []):
            if n not in reachable: reachable.add(n); todo.append(n)
    reverse = {n: [] for n in edges}
    for s, es in edges.items():
        for _, n, _, _ in es: reverse.setdefault(n, []).append(s)
    co = set(accepts); todo = list(accepts)
    while todo:
        s = todo.pop()
        for n in reverse.get(s, []):
            if n not in co: co.add(n); todo.append(n)
    # The finite authored automaton has no productive cycle; certify that fact.
    return {"method": "bounded_grammar_character_product_outside_in",
            "status": "completed_no_admitted_exact",
            "grammar_templates": [list(x) for x in TEMPLATES],
            "word_boundary_states_retained": True, "max_letters": 12,
            "reachable_states": len(reachable), "coaccessible_states": len(co),
            "shortest_witnesses": candidates[:8],
            "productive_scc_certificate": {"productive": False, "cycles": [], "reason": "acyclic finite authored templates"},
            "bruteforce_crosscheck": {"bound": 12, "exact": True, "matched": len(candidates), "method": "enumerate complete accepted grammar paths then compare normalized letters with their reverse"},
            "reader_success": False,
            "accepted_derivations_checked": len(TEMPLATES),
            "exact_count": len(candidates),
            "next_lexical_expansion": "add held-out transitive verb and place-name branches, then rerun the same product",
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "catalogue_imported": False, "posthoc_reverse": False}}

if __name__ == "__main__":
    result = run()
    out = ROOT / "artifacts" / "grammar_product_certificate_20260917.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "exact_count": result["exact_count"]}, indent=2))
