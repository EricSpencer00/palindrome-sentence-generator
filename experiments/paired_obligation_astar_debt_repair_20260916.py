"""Concrete repair: carry one exposed character mismatch as bounded debt.

The base route prunes on the first mismatch. This repair changes construction
by allowing one mismatch to remain live while later grammar obligations are
realized, but still admits only an exact terminal tape. Partial renders remain
probes, never candidates.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.paired_obligation_astar_20260916 import run as base_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/paired-obligation-astar-debt-repair-20260916.json"
EXPERIMENT_ID = "paired-obligation-astar-debt-repair-20260916"
SIGNATURE = "paired-grammar-obligation-astar|bounded-character-mismatch-debt|deferred-independent-terminal-realization|semantic-role-completion-frontier|ordinary-order-complete-sentences|independent-two-pointer-audit"


def run(*, state_limit: int = 50_000) -> dict:
    result = base_run(state_limit=state_limit, mismatch_budget=1)
    result.update({
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_paired_obligation_astar_debt_repair",
        "repair_of": "paired-obligation-astar-20260916",
        "repair": "carry at most one exposed mismatch as construction debt; exact terminal closure remains mandatory",
        "novelty_preflight": {"registry_entries_before_run": 98, "excluded_routes_before_run": 6, "signature_overlap": ["paired-obligation-astar-20260916"], "manual_review_required": True, "disposition": "concrete construction repair; bounded debt changes admissible frontier, not just budget or seed"},
        "provenance": {**result["provenance"], "repair_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "programmatic_readability_claim": False},
    })
    return result


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
