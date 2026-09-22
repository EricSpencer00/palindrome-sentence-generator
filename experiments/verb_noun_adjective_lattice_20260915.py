"""Small concurrent V-N-ADJ grammar lattice.

The branch is intentionally construction-first: each side is a complete parse
and the two sides are joined only when their letter tapes cancel exactly.
"""
from __future__ import annotations

import hashlib
import json
import sys
from itertools import permutations, product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from experiments.luna_broad_grammar_probe_20260915 import Plan, lexical_table, pools, search_pair
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

SHAPES = tuple(permutations(("VERB", "NOUN", "ADJ")))
MIN_LETTERS = 39


def repair_operator(row: dict) -> dict:
    """Concrete next move for an exact closure rejected by a hard gate."""
    failed = [name for name, passed in row.get("mechanical_checks", {}).items() if not passed]
    return {"operator": "replace_one_lexeme", "failed_gates": failed,
            "action": "replace one content slot with the next ranked word of the same POS while preserving the residual tape"}


def run(pool_size: int = 180, budget: int = 40_000) -> dict:
    role_pools = pools(lexical_table(), pool_size)
    records = []
    stats = []
    for left, right in product(SHAPES, repeat=2):
        # search_pair's emitted right words are reversed for exact reverse-tape
        # joining; requiring both complete shapes gives a full parse witness.
        rows, info = search_pair(Plan("-".join(left), left),
                                 Plan("-".join(right), right),
                                 role_pools, budget)
        stats.append({"left": left, "right": right, **info, "closures": len(rows)})
        for row in rows:
            row["complete_parse"] = len(row["left_words"]) == len(left) and len(row["right_words"]) == len(right)
            row["reverse_tape_join"] = normalize_letters(" ".join(row["left_words"])) == normalize_letters(" ".join(row["right_words"]))[::-1]
            row["mechanical_checks"] = mechanical_admission_checks(row["text"], min_letters=MIN_LETTERS, max_letters=180)
            row["mechanically_eligible"] = row["complete_parse"] and row["reverse_tape_join"] and all(row["mechanical_checks"].values())
            if not row["mechanically_eligible"]:
                row["repair"] = repair_operator(row)
            records.append(row)
    unique = {r["text"]: r for r in records}
    eligible = [r for r in unique.values() if r["mechanically_eligible"]]
    return {"status": "verb_noun_adjective_lattice_complete", "shapes": [list(s) for s in SHAPES],
            "pool_size": pool_size, "min_letters": MIN_LETTERS, "exact_closures": len(unique),
            "mechanically_eligible": eligible, "records": list(unique.values()), "searches": stats,
            "repair_operator": repair_operator({}),
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "lexicon": "Brown POS + wordfreq"}}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); p.add_argument("--pool-size", type=int, default=180)
    args = p.parse_args()
    if args.out.exists(): p.error("refusing to overwrite output")
    result = run(args.pool_size)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"exact": result["exact_closures"], "eligible": len(result["mechanically_eligible"]) }))
