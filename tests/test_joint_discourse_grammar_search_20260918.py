import json
from pathlib import Path

from experiments.joint_discourse_grammar_search_20260918 import audit, search


RUN = Path(__file__).parents[1] / "runs/joint-discourse-grammar-search-20260918.json"


def test_rendered_run_has_unique_joint_candidates_and_provenance():
    data = json.loads(RUN.read_text())
    assert data["candidate_count"] == len(data["candidates"]) >= 1
    assert data["unique_sweeps"] > data["candidate_count"]
    rendered = [row["rendered"] for row in data["candidates"]]
    assert len(rendered) == len(set(rendered))
    for row in data["candidates"]:
        assert "?" in row["rendered"] and row["rendered"].endswith(".")
        assert row["provenance"] == "bounded_joint_live_tape_product"
        assert row["audit"]["forbidden_clear"]
        assert row["audit"]["exact"] == row["audit"]["independent_two_pointer"]
        assert len(row["audit"]["sha256"]) == 64


def test_audit_rejects_proper_palindrome_and_word_mirror():
    bad = audit("Did Ada see? Ada see.")
    assert bad["proper_palindromic_subspans"]
    assert not bad["forbidden_clear"]


def test_search_is_bounded_and_repeatable():
    a, b = search(5), search(5)
    assert [r["audit"]["sha256"] for r in a["candidates"]] == [r["audit"]["sha256"] for r in b["candidates"]]
