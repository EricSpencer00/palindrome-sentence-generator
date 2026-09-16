"""Concrete repair: permit a palindromic character-centre residual.

The parent center-out POS run incorrectly required zero debt after the final
word slots.  This repair keeps the same lexicalization state but accepts only
a residual that is itself a palindrome, which is the exact condition for an
odd-length character tape whose centre lies inside a word.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments" / "pos_template_centerout_longform_repair_20260915.py"
spec = importlib.util.spec_from_file_location("parent_centerout", PARENT)
assert spec and spec.loader
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)

EXPERIMENT_ID = "pos-template-centerout-longform-center-residual-repair-20260915"
SIGNATURE = "pos-template-centerout-longform-center-residual-repair|odd-palindromic-center-debt|debt-carrying-character-equation|variable-word-boundary|independent-tape-audit"


def run() -> dict:
    by = parent._brown_lexicon()
    rows = []
    stats = {}
    for name, tags in parent.TEMPLATES.items():
        found, st = parent.solve(tags, by, budget=1_200_000)
        for row in found:
            row["template_name"] = name
        rows.extend(found)
        for key, value in st.items():
            stats[f"{name}.{key}"] = value
    unique = {}
    for row in rows:
        key = row["audit"]["normalized_tape"]
        if key not in unique or row["lm_prior"] > unique[key]["lm_prior"]:
            unique[key] = row
    rendered = sorted(unique.values(), key=lambda r: (r["mechanically_admitted"], r["lm_prior"], r["audit"]["letters"]), reverse=True)
    admitted = [r for r in rendered if r["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_no_reader_promotion" if not admitted else "exact_hits_pending_blinded_readers",
        "parent_experiment": "pos-template-centerout-longform-repair-20260915",
        "repair": "allow only a palindromic residual at the final character centre",
        "config": {"templates": {k: list(v) for k, v in parent.TEMPLATES.items()}, "inventory_sizes": {k: len(v) for k, v in by.items()}, "budget_per_template": 1_200_000, "catalogue_text_copied": False},
        "stats": {**stats, "unique_terminal_rows": len(rendered), "mechanically_admitted": len(admitted), "reader_eligible": 0},
        "rendered_candidates": rendered[:100],
        "exact_candidates": admitted,
        "next_repair": "At the best exact seam, substitute a held-out agreement-compatible lexical item and re-run the same centre-residual audit.",
        "provenance": {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "parent_script_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(), "programmatic_readability_claim": False},
    }


if __name__ == "__main__":
    out = run()
    path = ROOT / "runs" / "pos-template-centerout-longform-center-residual-repair-20260915.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "stats": out["stats"], "path": str(path)}, indent=2))
