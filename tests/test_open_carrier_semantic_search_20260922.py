import hashlib
import json
from pathlib import Path

from experiments.open_carrier_semantic_search_20260922 import OUT, RAW, run


ROOT = Path(__file__).parents[1]


def test_longer_frontier_is_exact_and_mechanically_clean_but_semantically_rejected():
    data = run()
    row = data["exact_longer_candidates"][0]
    assert row["rendered"] == (
        "Nine poll assert. Spot spoons. Snoop. Stop. Stress all open in."
    )
    assert row["independent_audit"]["letters"] == 48
    assert row["independent_audit"]["two_pointer_exact"] is True
    assert row["independent_audit"]["hashes_agree"] is True
    assert row["carrier_equation"]["holds"] is True
    assert row["carrier_equation"]["owner_at_center_entry"] == "right"
    assert row["carrier_equation"]["residual_at_center_entry"] == "s"
    assert row["mechanically_admitted"] is True
    assert row["boundary_mask"]["passed"] is True
    assert row["semantic_status"] == "rejected_unreadable_pos_ambiguity"
    assert data["accepted_improvements"] == []


def test_live_trace_preserves_owner_residual_and_exact_cursors_to_closure():
    for row in (run()["baseline"], run()["exact_longer_candidates"][0]):
        trace = row["live_owner_residual_cursor_trace"]
        assert trace
        assert all(step["owner_after"] in {"left", "right", "none"} for step in trace)
        assert all(step["left_word_cursor"] >= 0 for step in trace)
        assert all(step["right_word_cursor"] >= -1 for step in trace)
        assert trace[-1]["owner_after"] == "none"
        assert trace[-1]["residual_after"] == ""


def test_remote_search_is_bounded_and_records_the_diverse_frontier():
    data = run()
    remote = data["remote_search"]
    assert remote["host"] == "hst-bench"
    assert remote["attested_span_stats"]["accepted_span_occurrences"] == 2_749_308
    assert remote["attested_span_stats"]["matches"] == 0
    assert remote["typed_template_stats"]["q_products_tested"] == 8_344_980
    assert remote["typed_template_stats"]["matches"] == 1
    assert {row["residual"] for row in data["diverse_residual_frontier"]} == {
        "s", "ac", "ca", "no", "on", "ow", "wo",
    }
    assert all(row["all_equations_hold"] for row in data["diverse_residual_frontier"])


def test_remote_artifacts_match_the_checked_in_probe_sources():
    data = run()
    names = {
        "brown": "open_carrier_brown_probe_20260922.py",
        "templates": "open_carrier_template_probe_20260922.py",
        "cycles": "open_cycle_word_probe_20260922.py",
    }
    for key, name in names.items():
        actual = hashlib.sha256((ROOT / "experiments" / name).read_bytes()).hexdigest()
        assert actual == data["provenance"]["remote_source_sha256"][key]
    assert all((ROOT / path).exists() for path in data["remote_search"]["raw_artifacts"])


def test_checked_in_summary_matches_replay():
    assert OUT.exists()
    assert json.loads(OUT.read_text()) == run()
