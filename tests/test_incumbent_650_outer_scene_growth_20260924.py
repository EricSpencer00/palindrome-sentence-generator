from __future__ import annotations

import re

from experiments.incumbent_650_outer_scene_growth_20260924 import (
    EXPECTED_SHA256,
    PARENT_SHA256,
    build_payload,
)


def test_fresh_outer_event_pair_grows_exact_650_lineage() -> None:
    payload = build_payload()
    row = payload["rows"][0]
    tape = re.sub(r"[^a-z]", "", row["rendered"].casefold())

    assert payload["parent"]["sha256"] == PARENT_SHA256
    assert row["parent_sha256"] == PARENT_SHA256
    assert row["audit"]["letters"] == 654
    assert payload["selection_policy"]["status"] == "unranked_readability_comparison_frontier"
    assert payload["selection_policy"]["length_role"].startswith("descriptive only")
    assert [row["letters"] for row in payload["comparison_frontier"]] == [650, 654]
    assert row["audit"]["sha256_forward"] == EXPECTED_SHA256
    assert tape == tape[::-1]
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["byte_pointer_exact"]
    assert row["audit"]["project_validator_exact"]


def test_new_scene_closes_residual_and_keeps_reader_gate_honest() -> None:
    row = build_payload()["rows"][0]
    replacement = row["replacement"]
    seam = row["live_seam"]

    assert replacement["new_event_content"] == [
        "Nadia stops a ram", "Mara spots Aidan"
    ]
    assert replacement["equation_exact"]
    assert seam["right_obligation"] == seam["right_consumption"]
    assert seam["final_residual"] == ""
    assert row["repair_debt"]["human_certified"] is False
    assert row["next_reader_facing_test"]["status"] == "not_administered"
    assert "reader ratings" in row["next_operator"]
