import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_repair_audits_all_families_and_preserves_near_misses():
    run = json.loads((ROOT / "runs/semantic-insertion-repair-20260916.json").read_text())
    assert run["novelty_preflight"]["registry_entries_read_before_run"] == 97
    assert run["novelty_preflight"]["registry_entries_after_run"] == 97
    assert run["exact_count"] == 0
    assert run["reader_eligible_count"] == 0
    assert run["near_miss_count"] == run["candidate_count"] == 12
    assert all(row["independent_span_grammar"] == [True, True] for row in run["candidates"])
    assert all(row["full_sentence_grammar"] is False for row in run["candidates"])
    assert all(row["reader_eligible"] is False for row in run["candidates"])
