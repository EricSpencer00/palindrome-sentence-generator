"""Variable-length repair over a role-aware reversible lexical reservoir.

The long fixed POS pattern exhausted its character debt before finding a
terminal.  This successor changes the grammar dimension itself: several
complete 5--10-slot sentence templates are solved independently, while the
corpus-derived reversible reservoir and palindromic centre rule remain fixed.
"""
from __future__ import annotations

import hashlib, json, importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments" / "role_aware_reversible_reservoir_centerout_20260915.py"
spec = importlib.util.spec_from_file_location("reservoir", PARENT)
assert spec and spec.loader
reservoir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reservoir)

EXPERIMENT_ID = "variable-length-role-reservoir-centerout-20260915"
SIGNATURE = "variable-length-role-reservoir-centerout|corpus-derived-reversible-lexicon|typed-semantics|palindromic-centre-residual|independent-audit"
TEMPLATES = {
    "svo_adv": ("DET", "NOUN", "VERB", "DET", "NOUN", "ADV"),
    "svo_pp": ("DET", "NOUN", "VERB", "PREP", "DET", "NOUN"),
    "event": ("PRON", "VERB", "DET", "ADJ", "NOUN", "ADV"),
    "report": ("NOUN", "PRON", "VERB", "DET", "NOUN", "ADP", "NOUN"),
    "complex": ("NOUN", "VERB", "PRON", "VERB", "DET", "NOUN", "ADP", "NOUN"),
    "long_event": ("NOUN", "VERB", "PRON", "VERB", "DET", "ADJ", "NOUN", "ADV", "ADP", "NOUN"),
}


def run() -> dict:
    by, reservoir_info = reservoir.reservoir(reservoir.parent._brown_lexicon())
    rows = []
    stats = Counter()
    for name, tags in TEMPLATES.items():
        found, state = reservoir.parent.solve(tags, by, budget=1_500_000)
        for row in found:
            row["template_name"] = name
        rows.extend(found)
        stats.update({f"{name}.{k}": v for k, v in state.items()})
    unique = {}
    for row in rows:
        tape = row["audit"]["normalized_tape"]
        if tape not in unique or row["lm_prior"] > unique[tape]["lm_prior"]:
            unique[tape] = row
    rendered = sorted(unique.values(), key=lambda r: (r["mechanically_admitted"], r["lm_prior"], r["audit"]["letters"]), reverse=True)
    admitted = [row for row in rendered if row["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_no_reader_promotion" if not admitted else "exact_hits_pending_blinded_readers",
        "parent_repair": "role-aware-reversible-reservoir-centerout-20260915",
        "repair": "replace one exhausted long POS shape with a deterministic variable-length template family",
        "config": {"templates": {k: list(v) for k, v in TEMPLATES.items()}, "inventory_sizes": {k: len(v) for k, v in by.items()}, "budget_per_template": 1_500_000, "catalogue_text_copied": False},
        "reservoir": reservoir_info,
        "stats": {**stats, "unique_terminal_rows": len(rendered), "mechanically_admitted": len(admitted), "reader_eligible": 0},
        "rendered_candidates": rendered[:100],
        "exact_candidates": admitted,
        "next_repair": "Pair distinct left/right semantic templates with agreement-bearing lexicalization; do not replay same-template or enlarge this reservoir.",
        "provenance": {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "parent_script_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(), "programmatic_readability_claim": False},
    }


if __name__ == "__main__":
    out = run()
    path = ROOT / "runs" / "variable-length-role-reservoir-centerout-20260915.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "stats": out["stats"], "path": str(path)}, indent=2))
