import json

from experiments.incumbent_666_causal_window_repair_20260922 import (
    CHILD_SHA256,
    NEW_LEFT,
    NEW_RIGHT,
    OUT,
    PARENT_SHA256,
    independent_audit,
    normalize,
)


def test_causal_windows_preserve_exact_666_lineage():
    payload = json.loads(OUT.read_text())
    row = next(
        row for row in payload["rows"] if row["id"] == "causal-window-repair-nadia-saw-666"
    )
    result = independent_audit(row["rendered"])

    assert row["parent_sha256"] == PARENT_SHA256
    assert result["normalized_letters"] == 666
    assert result["two_pointer_exact"]
    assert result["sha256_forward"] == CHILD_SHA256
    assert result["sha_equal"]
    assert normalize(NEW_LEFT) == normalize(NEW_RIGHT)[::-1]
    assert len(normalize(NEW_LEFT)) == 57
    assert len(normalize(NEW_RIGHT)) == 57
    assert row["live_seam"]["final_residual"] == ""
    assert row["live_seam"]["committed_character_contradictions"] == 0


def test_causal_windows_remove_targeted_fragments_and_keep_complete_clauses():
    payload = json.loads(OUT.read_text())
    row = next(
        row for row in payload["rows"] if row["id"] == "causal-window-repair-nadia-saw-666"
    )
    rendered = row["rendered"]

    assert "Aron, spam. Eh, but a star spots Aram. Spam's reviled, Nadia" not in rendered
    assert "Aidan delivers maps. Mara stops rats. A tub? He maps Nora" not in rendered
    for clause in (
        "Nadia saw Noel live.",
        "Mara stops Nadia.",
        "Nora sees Aram.",
        "Sara saw Noel live.",
        "Evil Leon was Aras.",
        "Mara sees Aron.",
        "Aidan spots Aram.",
        "Evil Leon was Aidan.",
    ):
        assert clause in rendered
    assert row["readability_delta"]["complete_finite_clauses"]
    assert not row["readability_delta"]["predicate_less_fragment"]

