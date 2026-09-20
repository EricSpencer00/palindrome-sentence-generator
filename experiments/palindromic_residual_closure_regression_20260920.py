"""Replay the corrected center rule on unchanged bounded lexical banks."""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bilateral_grammar_csp_20260920 import _consume, palindromic_residual
from experiments.semantic_role_bilateral_20260920 import run as semantic_run
from experiments.lexicalized_constituent_interior_csp_20260920 import run as constituent_run


def pointer_audit(text):
    tape = "".join(c.lower() for c in text if c.isascii() and c.isalpha())
    lo, hi = 0, len(tape) - 1
    while lo < hi:
        if tape[lo] != tape[hi]:
            return {"exact": False, "letters": len(tape), "first_mismatch": [lo, hi]}
        lo += 1
        hi -= 1
    return {"exact": bool(tape), "letters": len(tape),
            "sha256_forward": sha256(tape.encode()).hexdigest(),
            "sha256_reverse": sha256(tape[::-1].encode()).hexdigest()}


def proof():
    words = ["".join(chars) for n in range(1, 5) for chars in product("ab", repeat=n)]
    old = corrected = oracle = disagreements = 0
    for left in words:
        for right in words:
            residual = _consume(left, right[::-1])
            actual = residual is not None and palindromic_residual(*residual)
            expected = pointer_audit(left + right)["exact"]
            old += residual == ("", "")
            corrected += actual
            oracle += expected
            disagreements += actual != expected
    assert not disagreements
    return {"fixture": "synthetic binary strings, never English candidate material",
            "pairs": len(words) ** 2, "old_empty_center_closures": old,
            "corrected_closures": corrected, "independent_oracle_closures": oracle,
            "disagreements": disagreements}


def run():
    result = {"experiment_id": "palindromic-residual-closure-regression-20260920",
              "change": "allow a symmetric unmatched middle after both grammar derivations finish",
              "host": platform.node(), "synthetic_proof": proof(), "banks": {},
              "reader_status": "not run; exactness is not readability evidence"}
    for name, function, bounds, key in (
        ("semantic_role", semantic_run, {"max_nodes": 3_000_000}, "paths"),
        ("constituent_interior", constituent_run, {"max_states": 120_000, "min_letters": 39}, "exact_candidates"),
    ):
        bank = function(**bounds)
        rows = bank[key]
        for row in rows:
            row["independent_pointer_audit"] = pointer_audit(row["rendered"])
            assert row["independent_pointer_audit"]["exact"]
        result["banks"][name] = {"bounds": bounds, "stats": bank["stats"],
                                 "exact_rows": rows,
                                 "exact_over_38": sum(row["independent_pointer_audit"]["letters"] > 38 for row in rows)}
    sources = ["bilateral_grammar_csp_20260920.py", "forward_lexicalized_grammar_20260920.py",
               "experiments/semantic_role_bilateral_20260920.py",
               "experiments/lexicalized_constituent_interior_csp_20260920.py", str(Path(__file__).relative_to(ROOT))]
    result["source_sha256"] = {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"proof": result["synthetic_proof"], "banks": {
        key: {"stats": bank["stats"], "exact_over_38": bank["exact_over_38"], "exact_rows": len(bank["exact_rows"])}
        for key, bank in result["banks"].items()}}, sort_keys=True))
