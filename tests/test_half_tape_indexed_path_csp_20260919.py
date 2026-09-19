from experiments.half_tape_indexed_path_csp_20260919 import audit, run, search


def test_seed_is_recovered_by_ordinary_path_search():
    result = search(38)
    admitted = [row["rendered"] for row in result["mechanically_admitted"]]
    assert "an aide rips nine memos some men inspire Diana." in admitted


def test_independent_audit_has_matching_digests():
    row = search(38)["mechanically_admitted"][0]
    checked = audit(row["rendered"])
    assert checked["two_pointer_exact"]
    assert checked["sha_equal"]
    assert checked["sha256_forward"] == checked["sha256_reverse"]


def test_bounded_run_reports_actual_rows_and_no_rlaif():
    result = run(range(38, 42), max_nodes=20_000)
    assert result["actual_candidates"]
    assert result["provenance"]["rlaif_per_candidate"] is False
    assert result["stats"]["mechanically_admitted"] >= 2
