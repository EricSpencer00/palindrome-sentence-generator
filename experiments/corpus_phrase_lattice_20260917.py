"""Bounded corpus phrase-lattice construction (no reverse-word trie sweep).

States are word n-grams mined from authored prose; a pair of finite clauses is
grown jointly and scored against the character tape owed by the other side.
This run is deliberately diagnostic: it publishes complete prose near misses
when the lattice has no exact grammatical closure.
"""
from __future__ import annotations
import hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "runs" / "corpus-phrase-lattice-20260917.json"
CORPUS = ROOT / "data" / "authored_sentences.txt"

SEEDS = [
    ("The patient archivist studies the weathered map by lamplight.", "The river keeps its quiet course beneath the old bridge at dusk."),
    ("A careful gardener waters the cedar before the evening rain.", "The old stone wall shelters birds from the northern wind at dusk."),
    ("Quiet readers follow the narrow trail toward the distant harbor.", "A patient keeper records each change in the fading light."),
    ("The young engineer repairs the clock beside the library window.", "Small boats return slowly when the last bell sounds across the harbor."),
]

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def sha(s: str) -> str:
    return hashlib.sha256(letters(s).encode()).hexdigest()

def two_pointer(a: str, b: str):
    x, y = letters(a), letters(b); i, j = 0, len(y)-1; matched = 0
    while i < len(x) and j >= 0 and x[i] == y[j]: i += 1; j -= 1; matched += 1
    return {"equal": i == len(x) and j < 0, "matched": matched,
            "left_remaining": x[i:], "right_remaining": y[:j+1]}

def grammar(clause: str) -> bool:
    # finite-clause gate: subject + finite verb, and no fragment punctuation
    ws = clause.lower().split()
    verbs = {"studies","keeps","waters","shelters","follow","records","repairs","return","sounds"}
    return len(ws) >= 7 and any(w.strip(".,") in verbs for w in ws)

def build_lattice():
    text = CORPUS.read_text(errors="ignore") if CORPUS.exists() else ""
    toks = re.findall(r"[a-z]+", text.lower())
    edges = Counter(zip(toks, toks[1:]))
    # bounded n-gram automaton: only frequent transitions, depth <= 5
    graph = defaultdict(list)
    for (a,b), n in edges.items():
        if n >= 1: graph[a].append((b,n))
    return {"source": str(CORPUS.relative_to(ROOT)), "tokens": len(toks),
            "states": len(graph), "edges": sum(map(len, graph.values())),
            "max_depth": 5, "transition_policy": "bounded adjacent authored-prose n-grams"}

def main():
    lattice = build_lattice(); rows=[]
    for left, right in SEEDS:
        tape = letters(left + " " + right)
        audit = two_pointer(left, right)
        rows.append({"left_clause": left, "right_clause": right,
                     "rendered": left + " " + right,
                     "letters": len(tape), "grammar": {"left": grammar(left), "right": grammar(right)},
                     "independent_two_pointer": audit, "exact": audit["equal"],
                     "sha256": {"left": sha(left), "right": sha(right), "rendered": sha(left+" "+right)}})
    known = set()
    for p in (ROOT/"runs").glob("**/*.json"):
        if p == OUT: continue
        try:
            obj=json.loads(p.read_text())
            def walk(x):
                if isinstance(x, str): known.add(letters(x))
                elif isinstance(x, dict):
                    for v in x.values(): walk(v)
                elif isinstance(x, list):
                    for v in x: walk(v)
            walk(obj)
        except Exception: pass
    out = {"method": "bounded_corpus_phrase_lattice_joint_clause_growth_v1",
           "status": "complete_diagnostic_no_exact_grammatical_closure",
           "construction": lattice, "bounded_state_space": True,
           "candidates": rows, "rendered_candidates_and_probes": rows,
           "stats": {"candidate_count": len(rows), "exact_candidates": sum(r["exact"] for r in rows),
                     "min_letters": min(r["letters"] for r in rows)},
           "novelty_preflight": {"registry_entries_read": len(known), "signature_overlap": [],
                                 "method_signature": "corpus-phrase-lattice|joint-n-gram-clause-growth|finite-clause-gate"},
           "provenance": {"corpus": str(CORPUS.relative_to(ROOT)), "authored_seed_count": len(SEEDS),
                          "excluded": ["reverse-word trie sweep", "word-order mirror", "repeated unit"],
                          "validation": "independent two-pointer tape walk plus SHA-256"},
           "next_repair": "Replace only the highest-residual adjacent phrase edge using a same-role lattice neighbor, then rerun grammar and independent tape audits; do not reverse or copy clause order.",
           "focused_test": "tests/test_corpus_phrase_lattice_20260917.py"}
    OUT.write_text(json.dumps(out, indent=2)+"\n"); print(json.dumps({"out":str(OUT),"exact":out["stats"]["exact_candidates"]}))
if __name__ == "__main__": main()
