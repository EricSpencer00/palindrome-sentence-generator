import json
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "minimal_residual_grammar_20260916",
    Path(__file__).parents[1] / "experiments/minimal_residual_grammar_20260916.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)
grow = MODULE.grow
independent_audit = MODULE.independent_audit
normalize = MODULE.normalize
residual = MODULE.residual
run = MODULE.run


def test_center_first_growth_keeps_complete_agreeing_clauses_and_live_debt():
    rows = grow()
    assert [row["depth"] for row in rows] == [1, 2, 3, 4]
    assert rows[-1]["letters"] > 100
    assert all(row["center_bridge"] == "e" for row in rows)
    assert all(row["feature_agreement"] for row in rows)
    assert all(not row["repeated_unit_rejected"] for row in rows)
    assert all(not row["self_palindromic_unit_rejected"] for row in rows)
    assert all(row["finished_tape_reversal_rejected"] for row in rows)
    assert all(row["residual"]["left_residual"] or row["residual"]["right_residual"] for row in rows)


def test_audit_and_residual_are_independent_of_finished_tape_reversal():
    text = grow()[-1]["text"]
    audit = independent_audit(text)
    assert audit["letters"] == len(normalize(text))
    assert audit["sha256_forward"] != audit["sha256_reverse"]
    debt = residual("e", "The pilot checks the engine", "The sailor marks the distant buoy")
    assert debt["center"] == "e"
    assert debt["matched_outer_pairs"] < 10
    assert not debt["closed"]


def test_run_artifact_records_provenance_and_no_exact_claim():
    data = run()
    assert data["exact_candidates"] == 0
    assert data["reader_eligible"] == []
    path = Path(__file__).parents[1] / "runs/minimal-residual-grammar-20260916.json"
    saved = json.loads(path.read_text())
    assert saved["provenance"]["generator_sha256"] == data["provenance"]["generator_sha256"]
    assert saved["novelty_preflight"]["status"] == "passed"
