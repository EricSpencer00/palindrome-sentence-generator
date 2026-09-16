from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.paired_obligation_astar_debt_repair_20260916 import run


def test_debt_repair_is_concrete_and_partial_probes_are_not_candidates():
    result = run(state_limit=2_000)
    assert result["repair_of"] == "paired-obligation-astar-20260916"
    assert result["config"]["mismatch_budget"] == 1
    assert result["stats"]["rendered_probes"] > 0
    assert result["stats"]["candidate_count"] == 0
    assert all(row["candidate"] is False for row in result["rendered_candidates_and_probes"])
    assert all(row["probe_status"] == "partial_completed_render" for row in result["rendered_candidates_and_probes"])
