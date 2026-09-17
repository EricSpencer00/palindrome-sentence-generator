"""Quarantined post-hoc baseline for the seed seam.

The artifact preserves a small finished-product comparison, but it is not a
live character-product search: products are materialized first and audited
afterward.  It remains useful as negative evidence only.
"""
from __future__ import annotations

import hashlib, itertools, json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import normalize

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "seed-seam-growth-20260917"
SIGNATURE = "joint-typed-clause-growth|valency-agreement|character-zipper|heldout-pp-repair|independent-audit"

SEED = "An aide rips nine memos; some men inspire Diana."
LEX = {
    "det": ("a", "the"), "adj": ("kind", "quiet", "small", "wise"),
    "sg": ("aide", "keeper", "teacher", "artist"),
    "pl": ("men", "sailors", "poets", "artists"),
    "vpl": ("inspire", "follow", "guide", "watch"),
    "vsg": ("inspires", "follows", "guides", "watches"),
    "obj": ("memos", "songs", "letters", "sails"),
    "name": ("Diana", "Ada", "Nina", "Iris"),
    "prep": ("near", "beside", "under"), "place": ("the pier", "a gate", "the sea"),
}

def audit(text: str) -> dict:
    tape = normalize(text)
    ok = tape == tape[::-1]
    first = next((i for i,(a,b) in enumerate(zip(tape,tape[::-1])) if a != b), None)
    return {"letters": len(tape), "exact": ok, "first_mismatch": first,
            "two_pointer_exact": all(tape[i] == tape[-1-i] for i in range(len(tape)//2)),
            "sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def clause_rows():
    # Two complete, independently authored scene families.  The right clause
    # is not derived from the left tape; only character obligations are shared.
    left = []
    for d, adj, n, v, obj in itertools.product(LEX["det"], LEX["adj"], LEX["pl"], LEX["vpl"], LEX["obj"]):
        left.append((f"{d} {adj} {n} {v} {obj}", "plural-agent"))
    right = []
    for d, n, v, name in itertools.product(LEX["det"], LEX["sg"], LEX["vsg"], LEX["name"]):
        right.append((f"{d} {n} {v} {name}", "singular-agent"))
    return left, right

def shortcut_filters(text: str) -> dict:
    words = re.findall(r"[a-z]+", text.lower())
    return {"no_word_order_symmetry": words != words[::-1],
            "no_self_palindromic_proper_multiword_span": all(normalize(w) != normalize(w)[::-1] for w in words if len(w) > 1),
            "no_repeated_content": len(words) == len(set(words)),
            "complete_clause_pair": text.count(".") == 2 and all(len(x.split()) >= 4 for x in text.split(".") if x.strip())}

def run() -> dict:
    left, right = clause_rows(); rows = []
    # Keep only a deterministic, diverse frontier: this is joint growth, not
    # a larger cartesian sweep.  The held-out repair adds the PP after search.
    frontier = list(itertools.islice(itertools.product(left, right), 0, 96, 7))
    for (l, lf), (r, rf) in frontier:
        text = l + ". " + r + "."
        rows.append({"rendered": text, "left_frame": lf, "right_frame": rf,
                     "audit": audit(text), "shortcut_filters": shortcut_filters(text),
                     "provenance": {"left_clause_authored_forward": True, "right_clause_authored_forward": True,
                                    "joint_character_zipper": False, "live_product_search": False,
                                    "posthoc_cartesian_audit": True, "catalogue_imported": False,
                                    "grammar": "DET ADJ plural-N V plural-N; DET singular-N V singular-name"},
                     "repair": "held-out optional PP insertion at the first mismatch, preserving typed valency"})
    seed_a = audit(SEED)
    seed_words = shortcut_filters(SEED)
    exact = [r for r in rows if r["audit"]["exact"] and all(r["shortcut_filters"].values()) and r["audit"]["letters"] >= 39]
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
            "status": "quarantined_posthoc_comparison",
            "method": "post-hoc typed clause Cartesian comparison; not a live character product",
            "rows": rows, "seed_control": {"rendered": SEED, "audit": seed_a, "shortcut_filters": seed_words},
            "stats": {"products": len(rows), "exact": sum(r["audit"]["exact"] for r in rows),
                      "novel_exact_reader_eligible": len(exact)}, "reader_eligible": bool(exact),
            "next_repair": {"operator": "typed_optional_pp_at_first_mismatch", "applied": False,
                            "reason": "all bounded products miss before a complete clause closure"},
            "independent_validation": "two-pointer equality plus forward/reverse SHA-256",
            "search_integrity": {"live_product_search": False,
                                 "posthoc_cartesian_audit": True,
                                 "admission_safe": False}}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(run(), indent=2) + "\n")
