import json
from pathlib import Path

RUN = Path(__file__).parents[1] / "runs/two-clause-joint-seam-bundle-search-20260917.json"

def test_rows_are_independently_audited_and_rendered():
    data = json.loads(RUN.read_text())
    assert data["candidate_count"] == len(data["candidates"]) > 0
    for row in data["candidates"]:
        assert row["audit"]["exact"] == row["audit"]["independent_two_pointer"]
        assert len(row["audit"]["sha256"]) == 64
        assert row["rendered"].startswith("The ")
        assert row["anti_shortcut"]["intact_prose"]
        assert row["anti_shortcut"]["repeated_unit"] == (row["left_bundle"] == row["right_bundle"] and row["left_setting"] == row["right_setting"])

def test_joint_rows_keep_three_word_agreement_bundles():
    data = json.loads(RUN.read_text())
    for row in data["candidates"]:
        assert len(row["left_bundle"]) == 3
        assert len(row["right_bundle"]) == 3
        assert row["provenance"] == "typed_two_clause_joint_seam_equation"
