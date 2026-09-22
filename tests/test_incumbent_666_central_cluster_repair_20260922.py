import json

from experiments.incumbent_666_central_cluster_repair_20260922 import (
    NEW_LEFT,
    NEW_RIGHT,
    OUT,
    CHILD_SHA256,
    PARENT_SHA256,
    independent_audit,
    normalize,
)


def test_central_repair_keeps_exact_666_lineage_and_closes_live_obligation():
    payload = json.loads(OUT.read_text())
    row = next(
        row for row in payload["rows"] if row["id"] == "central-cluster-repair-nadia-sees-666"
    )
    audit = independent_audit(row["rendered"])

    assert row["parent_sha256"] == PARENT_SHA256
    assert audit["normalized_letters"] == 666
    assert audit["two_pointer_exact"]
    assert audit["sha256_forward"] == CHILD_SHA256
    assert audit["sha_equal"]
    assert row["live_seam"]["final_residual"] == ""
    assert row["live_seam"]["committed_character_contradictions"] == 0
    assert normalize(NEW_LEFT) == normalize(NEW_RIGHT)[::-1]


def test_central_formula_is_replaced_by_complete_finite_clauses():
    payload = json.loads(OUT.read_text())
    row = next(
        row for row in payload["rows"] if row["id"] == "central-cluster-repair-nadia-sees-666"
    )
    rendered = row["rendered"]

    assert "Nadia delivers maps. Leon. Ari delivers maps." not in rendered
    assert "Spam's reviled, Ira. Noel; spam's reviled, Aidan" not in rendered
    for clause in (
        "Nadia sees Mara.",
        "Nora sees Nadia.",
        "Ari sees God.",
        "Dog sees Ira.",
        "Aidan sees Aron.",
        "Aram sees Aidan.",
    ):
        assert clause in rendered
    assert row["readability_delta"]["complete_finite_clauses"]
    assert not row["readability_delta"]["predicate_less_fragment"]
