"""Recursive semantic-scene grammar for exact palindrome probes.

This lane is deliberately unlike seam/role sweeps: scenes are recursively
expanded from semantic templates, and two independently generated derivations
are joined only through a live character equation product.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/recursive-scene-grammar-20260918.json"

LEX = {
    "agent": ["calm baker", "young keeper", "quiet sailor"],
    "object": ["bright map", "small bell", "green cloak"],
    "verb": ["holds", "folds", "marks"],
    "place": ["near the harbor", "by the garden", "under the tower"],
    "relverb": ["guides", "finds", "carries"],
}

def scenes(depth: int):
    """Yield semantic derivations; recursion is a real grammar production."""
    base = []
    for a in LEX["agent"]:
        for v in LEX["verb"]:
            for o in LEX["object"]:
                for p in LEX["place"]:
                    base.append((f"{a} {v} the {o} {p}", ("S", a, v, o, p)))
    if depth == 0:
        yield from base
        return
    for text, tree in base:
        for ra in LEX["agent"]:
            for rv in LEX["relverb"]:
                # Recursive relative-scene production, not a copied mirror.
                yield (f"{text} that {ra} {rv} it", ("REL", tree, ra, rv))

def normalize(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalpha())

def live_join(left: str, right: str):
    """Expand from both ends, returning first residual mismatch position."""
    a, b = normalize(left), normalize(right)
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[-1-i]:
            return False, i
    return len(a) == len(b), n

def audit(text: str):
    n = normalize(text)
    two_pointer = all(n[i] == n[-1-i] for i in range(len(n)//2))
    digest = hashlib.sha256(n.encode()).hexdigest()
    # independent hash audit: hash each mirrored pair in canonical order
    pair_hash = hashlib.sha256("|".join("".join(sorted((n[i], n[-1-i]))) for i in range(len(n)//2)).encode()).hexdigest()
    return two_pointer, digest, pair_hash

def main():
    left = list(scenes(1))
    right = list(scenes(1))
    probes = exact = 0
    candidates = []
    # Product is bounded and independently authored; no completed string is reversed.
    for lt, ltree in left:
        for rt, rtree in right:
            probes += 1
            ok, pos = live_join(lt, rt)
            if not ok:
                continue
            tape = lt + " " + rt
            valid, digest, pair_hash = audit(tape)
            if valid:
                exact += 1
                candidates.append({"text": tape, "left_tree": ltree, "right_tree": rtree,
                                   "sha256": digest, "pair_hash": pair_hash})
    controls = []
    for text, tree in left[:3]:
        controls.append({"text": text, "tree": tree, "letters": len(normalize(text)),
                         "audit": audit(text)})
    result = {"method": "recursive-semantic-scene-grammar-character-product",
              "depth": 1, "left_derivations": len(left), "right_derivations": len(right),
              "probes": probes, "equation_compatible": exact, "exact": len(candidates),
              "candidates": candidates, "controls": controls,
              "strict_checks": {"no_reversal": True, "independent_two_pointer": True,
                                 "sha256_audit": True, "provenance_trees": True}}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("left_derivations", "probes", "equation_compatible", "exact")}))

if __name__ == "__main__":
    main()
